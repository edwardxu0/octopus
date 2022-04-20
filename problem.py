import torch
import os
import numpy as np
import subprocess
import sys


import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

from octopus.artifact.artifacts import *

from ..plot.train_progress import ProgressPlot
from ..architecture.VeriNet import VeriNet


class Problem:
    def __init__(self, settings):
        self.override = settings.override
        self.sub_dirs = settings.sub_dirs
        self.cfg_train = settings.cfg_train
        self.cfg_heuristic = settings.cfg_heuristic
        self.cfg_verify = settings.cfg_verify
        self.logger = settings.logger
        self._setup_(settings)

    def _setup_(self, settings):
        self.seed = settings.seed
        torch.manual_seed(settings.seed)
        np.random.seed(settings.seed)
        torch.set_printoptions(threshold=100000)

        self.model_name = self.get_model_name(self.cfg_train, self.cfg_heuristic, self.seed)
        self.model_path = os.path.join(self.sub_dirs['model_dir'], f"{self.model_name}.onnx")

        # setup train data collectors
        self.train_stable_ReLUs = []
        self.train_BS_points = []
        self.train_loss = []
        self.test_accuracy = []

        cfg = self.cfg_train
        use_gpu = False if 'gpu' not in cfg else cfg['gpu']
        use_cuda = use_gpu and torch.cuda.is_available()
        amp = False if 'amp' not in cfg else cfg['amp']
        self.amp = use_cuda and amp
        if use_cuda:
            self.logger.info('CUDA enabled.')
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.artifact = globals()[cfg['artifact']](data_mode=cfg['adv_train'],
                                                   batch_size=cfg['batch_size'],
                                                   test_batch_size=cfg['test_batch_size'],
                                                   use_cuda=self.device)

        self.train_loader, self.test_loader = self.artifact.get_data_loader()

        self.model = VeriNet(self.artifact, self.cfg_train['net_layers'],
                             self.logger, self.device, self.amp).to(self.device)
        self.logger.info(f"Network:\n{self.model}")
        self.model._setup_heuristics(self.cfg_heuristic)

    @staticmethod
    def get_model_name(cfg_train, cfg_heuristic, seed):
        name = f"A={cfg_train['artifact']}"
        name += f"_N={cfg_train['net_name']}"
        name += f"_RE={cfg_train['ReLU_estimation']}"

        for h in cfg_heuristic:
            x = cfg_heuristic[h]

            if h == 'bias_shaping':
                if x['mode'] == 'standard':
                    m = 'S'
                else:
                    raise NotImplementedError
                name += f"_BS={m}:{x['intensity']}:{x['occurrence']}:{x['start']}:{x['end']}"

            elif h == 'rs_loss':
                if x['mode'] == 'standard':
                    m = 'S'
                else:
                    raise NotImplementedError
                name += f"_RS={m}:{x['weight']}:{x['start']}:{x['end']}"

            elif h == 'prune':
                if x['mode'] == 'structure':
                    m = 'S'
                else:
                    raise NotImplementedError

                if 're_arch' not in x:
                    r = '-'
                elif x['re_arch'] == 'standard':
                    r = 'S'
                else:
                    raise NotImplementedError

                name += f"_PR={m}:{r}:{x['sparsity']}:{x['start']}:{x['end']}"

            else:
                assert False
        name += f'_seed={seed}'
        return name

    #initialize the mask
    def get_mask(self):
        mask={}

        for name, module in self.model.filtered_named_modules:

            if 'conv' in name:
                mname= 'model.'+upname+'.out_channels'
                j=module.out_channels
                mask[name]=torch.ones(j)
            elif 'FC' in name:
                j=module.out_features
                mask[name]=torch.ones(j)
            

        return mask

    # Training ...

    def _trained(self):
        # TODO: account for log file and # epochs
        return os.path.exists(self.model_path)

    def train(self):
        if self._trained() and not self.override:
            self.logger.info('Skipping trained network.')
            return

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg_train['lr'])
        self.LR_decay_scheduler = StepLR(self.optimizer, step_size=1, gamma=self.cfg_train['gamma'])

        if self.amp:
            self.logger.info('CUDA AMP enabled.')
            self.amp_scaler = torch.cuda.amp.GradScaler()

        mask=self.get_mask()

        for epoch in range(1, self.cfg_train['epochs'] + 1):
            # H: pruning
            if 'prune' in self.cfg_heuristic\
                    and self.cfg_heuristic['prune']['start'] <= epoch <= self.cfg_heuristic['prune']['end']:
                data = [activation, mask]
                self.model.run_heuristics('prune', data)

            self._train_epoch(epoch)
            activation=self.model.register_activation_hocks()
            self._test_epoch(epoch)

            

            # save model
            if self.cfg_train['save_model']:
                dummy_input = torch.randn([1] + self.artifact.input_shape, device=self.device)
                torch.onnx.export(self.model, dummy_input, self.model_path, verbose=False)
                torch.save(self.model.state_dict(), f"{self.model_path[:-5]}.pt")

                if self.cfg_train['save_intermediate']:
                    torch.onnx.export(self.model, dummy_input, f"{self.model_path[:-5]}.{epoch}.onnx", verbose=False)
                    torch.save(self.model.state_dict(), f"{self.model_path[:-5]}.{epoch}.pt")

            self.LR_decay_scheduler.step()

            self._plot_train()

    def _train_epoch(self, epoch):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Use AMP
            # equivalent to:
            # with torch.cuda.amp.autocast(enabled=self.amp):
            # to disable warning when cuda==false and amp==true.
            with torch.autocast(self.device.type, enabled=self.amp):

                output_pre_softmax = self.model(data)
                output = F.log_softmax(output_pre_softmax, dim=1)

                loss = F.nll_loss(output, target)

                # H: RS loss
                if 'rs_loss' in self.cfg_heuristic\
                        and self.cfg_heuristic['rs_loss']['start'] <= epoch <= self.cfg_heuristic['rs_loss']['end']:

                    rs_loss = self.model.run_heuristics('rs_loss', data)
                    loss += rs_loss * self.cfg_heuristic['rs_loss']['weight']

                if loss.isnan():
                    raise ValueError('Loss is NaN.')

                if self.amp:
                    self.amp_scaler.scale(loss).backward()
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            # H: bias shaping
            if 'bias_shaping' in self.cfg_heuristic\
                    and batch_idx != 0\
                    and self.cfg_heuristic['bias_shaping']['start'] <= epoch <= self.cfg_heuristic['bias_shaping']['end']:

                # print('before', self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation']), self.test_loader)
                if self.model.run_heuristics('bias_shaping', data):
                    BS_point = len(self.train_loader) * (epoch-1) + batch_idx
                    self.train_BS_points += [BS_point]
                # print('after', self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation']), self.test_loader)

            if batch_idx % self.cfg_train['log_interval'] == 0:
                batch_stable_ReLU = self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation'], self.test_loader)
                self.logger.info(
                    f'[Train] epoch: {epoch} batch: {batch_idx:5} {100.*batch_idx/len(self.train_loader):5.2f}% Loss: {loss.item():10.6f} SR: {batch_stable_ReLU:5}')
            else:
                batch_stable_ReLU = self.train_stable_ReLUs[-1]

            self.train_stable_ReLUs += [batch_stable_ReLU]
            self.train_loss += [loss.item()]

    def _test_epoch(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output = F.log_softmax(output, dim=1)

                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_accuracy = correct / len(self.test_loader.dataset)
        self.test_accuracy += [test_accuracy]
        batch_stable_ReLU = self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation'], self.test_loader)
        self.logger.info(
            f'[Test] epoch: {epoch} loss: {test_loss:10.6f}, accuracy: {test_accuracy*100:.2f}% SR: {batch_stable_ReLU:5}\n')

    def _plot_train(self):
        # draw training progress plot
        X1 = range(len(self.train_stable_ReLUs))
        Y1 = self.train_stable_ReLUs
        X2 = self.train_BS_points
        Y2 = np.array(Y1)[self.train_BS_points]
        X3 = (np.array(range(len(self.test_accuracy)))+1)*len(self.train_loader)
        Y3 = self.test_accuracy
        X4 = range(len(self.train_loss))
        Y4 = self.train_loss

        max_safe_relu = sum([self.model.activation[layer].view(
            self.model.activation[layer].size(0), -1).shape[-1] for layer in self.model.activation])

        p_plot = ProgressPlot()
        p_plot.draw_train(X1, Y1, X2, Y2, (0, max_safe_relu))
        p_plot.draw_accuracy(X3, Y3, X4, Y4, (0, 1))

        title = f'# {self.model_name}'
        path = os.path.join(self.sub_dirs['figure_dir'], self.model_name+'.png')
        p_plot.save(title, path)
        p_plot.clear()

    # Verification ...

    def _verified(self, prop, eps, verifier):
        assert self._trained()
        # TODO: account for verification completion
        self.veri_log_path = os.path.join(self.sub_dirs['veri_log_dir'],
                                          f"{self.model_name}_P={prop}_E={eps}_V={verifier}.txt")
        return os.path.exists(self.veri_log_path)

    def verify(self):
        cfg = self.cfg_verify
        prop = cfg['property']
        eps = cfg['epsilon']
        verifier = cfg['verifier']
        debug = cfg['debug'] if 'debug' in cfg else None
        save_log = cfg['save_log'] if 'save_log' in cfg else None

        if self._verified(prop, eps, verifier) and save_log and not self.override:
            self.logger.info('Skipping verified problem.')
        else:

            # TODO: this is temp to verify the best acc epoch, fix later
            model_path = self.model_path
            train_log_path = os.path.join(self.sub_dirs['result_dir'], 'train_log', self.model_name+'.txt')
            # print(train_log_path)
            lines = [x.strip() for x in open(train_log_path, 'r').readlines() if ' [Test] 'in x]
            acc_train = np.array([float(x.split()[-3][:-1]) for x in lines])
            acc_relu = np.array([float(x.split()[-1]) for x in lines])
            assert acc_train.shape[0] == 100 and acc_relu.shape[0] == 100
            best_epoch_acc_train = np.argmax(acc_train) + 1
            best_epoch_acc_relu = np.argmax(acc_relu) + 1
            self.logger.debug(f"acc_train:, {acc_train}, argmax:, {np.argmax(acc_train)}, best_epoch:, {best_epoch_acc_train}")
            self.logger.debug(f"acc_relu:, {acc_relu}, argmax: {np.argmax(acc_relu)}, best_epoch, {best_epoch_acc_relu}")
            model_path = f'{self.model_path[:-5]}.{best_epoch_acc_train}.onnx'
            assert os.path.exists(model_path)
            self.logger.debug(f'Best model: {model_path}')

            # generate property
            self.logger.info('Generating property ...')
            prop_path = self.artifact.gen_property(prop, eps, self.sub_dirs['property_dir'])

            # compose DNNV command
            res_monitor_path = os.path.join(os.environ['DNNV'], 'tools', 'resmonitor.py')
            cmd = f"python3 {res_monitor_path} -T {cfg['time']} -M {cfg['memory']}"
            cmd += f" ./tools/run_DNNV.sh {prop_path} -N N {model_path}"

            if 'eran' in verifier:
                cmd += f" --eran --eran.domain {verifier.split('_')[1]}"
            else:
                cmd += f" --{verifier}"

            if debug:
                cmd += ' --debug'

            if save_log:
                veri_log_file = open(self.veri_log_path, 'w')
            else:
                veri_log_file = sys.stdout

            self.logger.info('Executing DNNV ...')
            self.logger.debug(cmd)

            sp = subprocess.Popen(cmd,
                                  shell=True,
                                  stdout=veri_log_file,
                                  stderr=veri_log_file)
            rc = sp.wait()
            assert rc == 0

    def analyze(self):
        # analyze train
        # analyze verification
        assert self._trained()
        assert self._verified(self.cfg_verify['property'],
                              self.cfg_verify['epsilon'],
                              self.cfg_verify['verifier'])
        self.logger.debug(f'Analyzing log: {self.veri_log_path}')
        veri_ans, veri_time = self.analyze_veri_log(self.veri_log_path)
        if veri_ans and veri_time:
            self.logger.info(f'Result: {veri_ans}, {veri_time}s.')
        else:
            self.logger.info(f'Failed.')

    @staticmethod
    def analyze_veri_log(log_path):
        lines = open(log_path, 'r').readlines()
        lines.reverse()
        veri_ans = None
        veri_time = None
        for l in lines:
            if '  result: ' in l:
                if 'Error' in l:
                    veri_ans = 'error'
                else:
                    veri_ans = l.strip().split()[-1]

            elif '  time: ' in l:
                veri_time = float(l.strip().split()[-1])

            elif ' Timeout' in l:
                veri_ans = 'timeout'
                veri_time = float(l.strip().split()[-3])
                break
            elif ' of memory' in l:
                veri_ans = 'memout'
                veri_time = float(l.strip().split()[-3])
                break

        return veri_ans, veri_time

    def _save_meta(self):
        ...