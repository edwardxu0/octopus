import torch
import os
import numpy as np
import subprocess
import sys
import logging


import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

from octopus.artifact.artifacts import *

from ..plot.train_progress import ProgressPlot
from ..architecture.ReLUNet import ReLUNet


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

        self.model_name = self.Utility.get_model_name(self.cfg_train, self.cfg_heuristic, self.seed)
        self.model_path = os.path.join(self.sub_dirs['model_dir'], f"{self.model_name}.onnx")
        self.train_log_path = os.path.join(self.sub_dirs['train_log_dir'],
                                           f"{self.model_name}.txt")

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
        self.model = ReLUNet(self.artifact, self.cfg_train['net_layers'],
                             self.logger, self.device, self.amp).to(self.device)
        self.logger.info(f"Network:\n{self.model}")
        self.logger.info(f"# ReLUs: {self.model.nb_ReLUs}")

    # Training ...
    def _setup_train(self):
        if 'save_log' in self.cfg_train and self.cfg_train['save_log']:
            self.logger.debug(f'Saving training log to: {self.train_log_path}')
            file_handler = logging.FileHandler(self.train_log_path, 'w')
            self.logger.addHandler(file_handler)

        # setup train data collectors
        self.train_stable_ReLUs = []
        self.train_BS_points = []
        self.train_loss = []
        self.test_accuracy = []

        # configure heuristics
        self.model._setup_heuristics(self.cfg_heuristic)

    def _trained(self):
        # TODO: account for log file and # epochs
        trained = True
        if 'save_log' in self.cfg_train and self.cfg_train['save_log']:
            if not os.path.exists(self.train_log_path):
                trained = False
            else:
                trained = len([x for x in open(self.train_log_path, 'r').readlines()
                              if '[Test] ' in x]) == self.cfg_train['epochs']
        if not os.path.exists(self.model_path):
            trained = False

        return trained

    def train(self):
        if self._trained() and not self.override:
            self.logger.info('Skipping trained network.')
        else:
            self._setup_train()

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg_train['lr'])
            self.LR_decay_scheduler = StepLR(self.optimizer, step_size=1, gamma=self.cfg_train['gamma'])

            if self.amp:
                self.logger.info('CUDA AMP enabled.')
                self.amp_scaler = torch.cuda.amp.GradScaler()

            for epoch in range(1, self.cfg_train['epochs'] + 1):
                self._train_epoch(epoch)
                self._test_epoch(epoch)

                # save model
                if self.cfg_train['save_model']:
                    self.logger.debug(f'Saving model: {self.model_path}')
                    dummy_input = torch.randn([1] + self.artifact.input_shape, device=self.device)
                    torch.onnx.export(self.model, dummy_input, self.model_path, verbose=False)
                    torch.save(self.model.state_dict(), f"{self.model_path[:-5]}.pt")

                    if self.cfg_train['save_intermediate']:
                        torch.onnx.export(self.model, dummy_input,
                                          f"{self.model_path[:-5]}.{epoch}.onnx", verbose=False)
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

                # [H] RS loss
                if 'rs_loss' in self.cfg_heuristic\
                    and self.Utility.heuristic_enabled_epochwise(epoch,
                                                                 self.cfg_heuristic['rs_loss']['start'],
                                                                 self.cfg_heuristic['rs_loss']['end']):

                    rs_loss = self.model.run_heuristics('rs_loss', data=data)
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

            # [H] bias shaping
            if 'bias_shaping' in self.cfg_heuristic\
                    and batch_idx != 0\
                    and self.Utility.heuristic_enabled_epochwise(epoch,
                                                                 self.cfg_heuristic['bias_shaping']['start'],
                                                                 self.cfg_heuristic['bias_shaping']['end']):

                # print('before', self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation']), self.test_loader)
                if self.model.run_heuristics('bias_shaping', data=data, epoch=epoch):
                    BS_point = len(self.train_loader) * (epoch-1) + batch_idx
                    self.train_BS_points += [BS_point]
                # print('after', self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation']), self.test_loader)

            if batch_idx % self.cfg_train['log_interval'] == 0:
                batch_stable_ReLU = self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation'], self.test_loader)
                relu_accuracy = batch_stable_ReLU/self.model.nb_ReLUs
                self.logger.info(
                    f'[Train] epoch: {epoch:3} batch: {batch_idx:5} {100.*batch_idx/len(self.train_loader):5.2f}% Loss: {loss.item():10.6f} SR: {relu_accuracy*100:.2f}%')
            else:
                relu_accuracy = self.train_stable_ReLUs[-1]

            self.train_stable_ReLUs += [relu_accuracy]
            self.train_loss += [loss.item()]

        # [H] pruning
        # prune after entire epoch trained
        # using pre-activation values of last mini-batch
        if 'prune' in self.cfg_heuristic\
                and self.Utility.heuristic_enabled_epochwise(epoch,
                                                             self.cfg_heuristic['prune']['start'],
                                                             self.cfg_heuristic['prune']['end']):
            self.model.run_heuristics('prune')

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
        relu_accuracy = batch_stable_ReLU / self.model.nb_ReLUs
        self.logger.info(
            f'[Test] epoch: {epoch:3} loss: {test_loss:10.6f}, accuracy: {test_accuracy*100:.2f}% SR: {relu_accuracy*100:.2f}%\n')

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

        # max_safe_relu = sum([self.model.activation[layer].view(
        #    self.model.activation[layer].size(0), -1).shape[-1] for layer in self.model.activation])

        p_plot = ProgressPlot()
        p_plot.draw_train(X1, Y1, X2, Y2, (0, 1))  # max_safe_relu))
        p_plot.draw_accuracy(X3, Y3, X4, Y4, (0, 1))

        title = f'# {self.model_name}'
        path = os.path.join(self.sub_dirs['figure_dir'], self.model_name+'.png')
        p_plot.save(title, path)
        p_plot.clear()

    # Verification ...
    def _setup_verification(self):
        target_model = self.cfg_verify['target_model'] if 'target_model' in self.cfg_verify else None

        if 'save_intermediate' not in self.cfg_train or not self.cfg_train['save_intermediate']\
                and target_model not in [None, 'last']:
            raise ValueError(
                '\'save_intermediate = true\' is required for verification model selection strategy other than \'last\' epoch.')

        target_epoch = self.Utility.get_target_epoch(target_model, self.train_log_path)

        self.veri_log_path = os.path.join(self.sub_dirs['veri_log_dir'],
                                          f"{self.model_name}_e={target_epoch}_{self.Utility.get_verification_postfix(self.cfg_verify)}.txt")

        return target_epoch

    def _verified(self):
        assert self._trained()
        # TODO: account for verification completion

        return os.path.exists(self.veri_log_path)

    def verify(self):
        cfg = self.cfg_verify
        prop = cfg['property']
        eps = cfg['epsilon']
        verifier = cfg['verifier']
        debug = cfg['debug'] if 'debug' in cfg else None
        save_log = cfg['save_log'] if 'save_log' in cfg else None

        target_epoch = self._setup_verification()

        if self._verified() and save_log and not self.override:
            self.logger.info('Skipping verified problem.')
        else:
            if 'save_intermediate' not in self.cfg_train or not self.cfg_train['save_intermediate']:
                model_path = self.model_path
            else:
                model_path = f'{self.model_path[:-5]}.{target_epoch}.onnx'
            self.logger.debug(f'Target model: {model_path}')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Missing model file: {model_path}')

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
        self._setup_verification()
        assert self._verified()
        self.logger.debug(f'Analyzing log: {self.veri_log_path}')
        veri_ans, veri_time = self.Utility.analyze_veri_log(self.veri_log_path)
        if veri_ans and veri_time:
            self.logger.info(f'Result: {veri_ans}, {veri_time}s.')
        else:
            self.logger.info(f'Failed.')

    def _save_meta(self):
        ...

    class Utility:
        @staticmethod
        def get_model_name(cfg_train, cfg_heuristic, seed):
            name = f"A={cfg_train['artifact']}"
            name += f"_N={cfg_train['net_name']}"
            name += f"_RE={cfg_train['ReLU_estimation']}"

            def parse_start_end(x):
                return str(x['start']).replace(' ', '')+':'+str(x['end']).replace(' ', '')

            for h in cfg_heuristic:
                x = cfg_heuristic[h]

                if h == 'bias_shaping':
                    if x['mode'] == 'standard':
                        m = 'S'
                    else:
                        raise NotImplementedError
                    d = '' if not 'decay' in x else f":{x['decay']}"

                    name += f"_BS={m}:{x['intensity']}:{x['occurrence']}{d}:{parse_start_end(x)}"

                elif h == 'rs_loss':
                    if x['mode'] == 'standard':
                        m = 'S'
                    else:
                        raise NotImplementedError
                    name += f"_RS={m}:{x['weight']}:{parse_start_end(x)}"

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

                    name += f"_PR={m}:{r}:{x['sparsity']}:{parse_start_end(x)}"

                else:
                    assert False
            name += f'_seed={seed}'
            return name

        @ staticmethod
        def get_verification_postfix(cfg_verify):
            p = cfg_verify['property']
            e = cfg_verify['epsilon']
            v = cfg_verify['verifier']
            return f'P={p}_E={e}_V={v}'

        @ staticmethod
        def heuristic_enabled_epochwise(epoch, start, end):
            assert type(start) == type(end)
            if isinstance(start, int):
                enabled = start <= epoch <= end
            elif isinstance(start, list):
                enabled = any([x[0] <= epoch <= x[1] for x in zip(start, end)])
            else:
                assert False
            return enabled

        @ staticmethod
        def get_target_epoch(target_model, train_log_path):
            lines = [x.strip() for x in open(train_log_path, 'r').readlines() if '[Test] ' in x]
            if not target_model or target_model == 'last':
                target_epoch = len(lines)
            elif target_model.startswith('best') or target_model.startswith('top'):
                acc_test = np.array([float(x.strip().split()[-3][:-1]) for x in lines])
                acc_relu = np.array([float(x.strip().split()[-1][:-1])/100 for x in lines])

                if target_model.startswith('best'):
                    if target_model == "best test accuracy":
                        target_epoch = np.argmax(acc_test)+1
                    elif target_model == "best relu accuracy":
                        target_epoch = np.argmax(acc_relu)+1
                    else:
                        assert False
                else:
                    threshold = float(target_model.split()[1])

                    if target_model.endswith('test accuracy'):
                        max_test = np.max(acc_test)
                        #print(acc_test, threshold, max_test)
                        candidates = np.where(acc_test >= max_test-threshold)
                        # print(candidates)
                        max_relu = np.max(acc_relu[candidates])
                        # print(max_relu)

                        candidates = np.where(acc_relu == max_relu)
                        # print(candidates)
                        max_test = np.max(acc_test[candidates])
                        # print(max_test)
                        set_relu = set(np.where(acc_relu == max_relu)[0].tolist())
                        set_test = set(np.where(acc_test == max_test)[0].tolist())
                        # print(set_relu)
                        # print(set_test)
                        final_candidates = set_relu.intersection(set_test)
                        assert len(final_candidates) == 1
                        target_epoch = final_candidates.pop() + 1

                    elif target_model.endswith('relu accuracy'):
                        max_relu = np.max(acc_relu)
                        candidates = np.where(acc_relu >= max_relu-threshold)
                        # print(candidates)
                        max_test = np.max(acc_test[candidates])
                        # print(max_relu)

                        candidates = np.where(acc_test == max_test)
                        # print(candidates)
                        max_relu = np.max(acc_relu[candidates])
                        # print(max_test)
                        set_test = set(np.where(acc_test == max_test)[0].tolist())
                        set_relu = set(np.where(acc_relu == max_relu)[0].tolist())
                        # print(set_relu)
                        # print(set_test)
                        final_candidates = set_test.intersection(set_relu)
                        assert len(final_candidates) == 1
                        target_epoch = final_candidates.pop() + 1
                        # print(target_epoch)
                    else:
                        assert False
            else:
                assert False, f'Unknown target_model: {target_model}'
            # print(target_epoch)
            return target_epoch

        @ staticmethod
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

                elif 'Timeout' in l:
                    veri_ans = 'timeout'
                    veri_time = float(l.strip().split()[-3])
                    break
                elif 'Out of Memory' in l:
                    veri_ans = 'memout'
                    veri_time = None  # float(l.strip().split()[-3])
                    break

            return veri_ans, veri_time
