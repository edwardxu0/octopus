from asyncio.log import logger
from asyncio.subprocess import STDOUT
import torch
import os
import numpy as np
import subprocess
import sys

import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

from octopus.artifact.artifacts import MNIST, FashionMNIST

from ..plot.train_progress import ProgressPlot
from ..architecture.architecture import VeriNet


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
        use_gpu = True if 'gpu' in cfg and cfg['gpu'] else False
        use_cuda = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.artifact = globals()[cfg['artifact']](cfg['adv_train'], cfg['batch_size'], cfg['test_batch_size'], self.device)
        self.train_loader, self.test_loader = self.artifact.get_data_loader()

        self.model = VeriNet(self.cfg_train['net_layers'], self.logger, self.device).to(self.device)
        self.logger.info(f"Network:\n{self.model}")
        self.model._setup_heuristics(self.cfg_heuristic)
    

    @staticmethod
    def get_model_name(cfg_train, cfg_heuristic, seed):
        name =  f"A={cfg_train['artifact']}"
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
    

    # Training ...
    def _trained(self):
        # TODO: account for log file and # epochs
        return os.path.exists(self.model_path)





    def train(self):
        if self._trained() and not self.override:
            self.logger.info('Skipping trained network.')
            return

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg_train['lr'])
        scheduler = StepLR(self.optimizer, step_size=1, gamma=self.cfg_train['gamma'])

        for epoch in range(1, self.cfg_train['epochs'] + 1):
            self._train_epoch(epoch)
            self._test_epoch(epoch)

            # H: pruning
            if 'prune' in self.cfg_heuristic\
                and self.cfg_heuristic['prune']['start'] <= epoch <= self.cfg_heuristic['prune']['end']:
                self.model.run_heuristics('prune', None)
            
            # save model
            if self.cfg_train['save_model']:
                dummy_input = torch.randn(1, 1, 28, 28, device=self.device)
                torch.onnx.export(self.model, dummy_input, self.model_path, verbose=False)
                torch.save(self.model.state_dict(), f"{self.model_path[:-4]}pt")
            
            scheduler.step()
        
            self._plot_train()


    def _train_epoch(self, epoch):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output_pre_softmax = self.model(data)
            output = F.log_softmax(output_pre_softmax, dim=1)

            loss = F.nll_loss(output, target)
            
            # H: RS loss
            if 'rs_loss' in self.cfg_heuristic\
                and self.cfg_heuristic['rs_loss']['start'] <= epoch <= self.cfg_heuristic['rs_loss']['end']:

                rs_loss = self.model.run_heuristics('rs_loss', data)
                loss += rs_loss * self.cfg_heuristic['rs_loss']['weight']
            
            loss.backward()
            self.optimizer.step()

            # H: bias shaping
            if 'bias_shaping' in self.cfg_heuristic\
                and batch_idx != 0\
                and self.cfg_heuristic['bias_shaping']['start'] <= epoch <= self.cfg_heuristic['bias_shaping']['end']:

                if np.random.rand() < self.cfg_heuristic['bias_shaping']['occurrence']:
                    # print('before', self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation']), self.test_loader)
                    if self.model.run_heuristics('bias_shaping', data):
                        BS_point = len(self.train_loader) * (epoch-1) + batch_idx
                        self.train_BS_points += [BS_point]
                    # print('after', self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation']), self.test_loader)

            
            if batch_idx % self.cfg_train['log_interval'] == 0:
                batch_stable_ReLU = self.model.estimate_stable_ReLU(self.cfg_train['ReLU_estimation'], self.test_loader)
                self.logger.info(f'[Train] epoch: {epoch} batch: {batch_idx:5} {100.*batch_idx/len(self.train_loader):5.2f} Loss: {loss.item():10.6f} SR: {batch_stable_ReLU:5}')
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

        self.logger.info(f'[Test] epoch: {epoch} loss: {test_loss:10.6f}, accuracy: {test_accuracy*100:.2f}%.\n')


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

        max_safe_relu = sum([self.model.activation[layer].view(self.model.activation[layer].size(0),-1).shape[-1] for layer in self.model.activation])
        
        p_plot = ProgressPlot()
        p_plot.draw_train(X1, Y1, X2, Y2, (0, max_safe_relu))
        p_plot.draw_accuracy(X3, Y3, X4, Y4, (0,1))

        title =  f'# {self.model_name}'
        path = os.path.join(self.sub_dirs['figure'], self.model_name+'.png')
        p_plot.save(title, path)
        p_plot.clear()


    # Verification ...
    def _verified(self, prop, eps, verifier):
        assert self._trained()
        # TODO: account for verification completion
        self.veri_log_path = os.path.join(self.sub_dirs['veri_log_dir'], f"{self.model_name}_P={prop}_E={eps}_V={verifier}.txt")
        return os.path.exists(self.veri_log_path)


    def verify(self):
        cfg = self.cfg_verify
        prop = cfg['property']
        eps = cfg['epsilon']
        verifier = cfg['verifier']
        debug = cfg['debug']
        save_log = cfg['save_log']

        if self._verified(prop, eps, verifier) and not self.override:
            self.logger.info('Skipping verified problem.')
        else:
            
            # generate property
            self.logger.info('Generating property ...')
            self.artifact.gen_property(prop, eps, self.sub_dirs['property_dir'])

            prop_path = os.path.join(self.sub_dirs['property_dir'],f'{prop}_{eps}.py')
            cmd = f"./tools/resmonitor.py -T {cfg['time']} -M {cfg['memory']}"
            cmd += f" ./tools/run_DNNV.sh {prop_path} -N N {self.model_path}"

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
                                shell = True,
                                stdout = veri_log_file,
                                stderr = veri_log_file)
            rc=sp.wait()
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
            
        return veri_ans, veri_time

    def _save_meta(self):
        ...