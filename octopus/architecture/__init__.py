from ..heuristic.prune import Prune
from ..heuristic.RS_loss import RSLoss
from ..heuristic.bias_shaping import BiasShaping
from torch import nn
import torch
import numpy as np
np.set_printoptions(suppress=True, precision=4)


class BasicNet(nn.Module):
    def __init__(self, logger, device, amp):
        super(BasicNet, self).__init__()
        self.logger = logger
        self.device = device
        self.amp = amp

    def __setup__(self):
        self._compute_filtered_named_modules()
        self.activation = self.register_activation_hocks()

    def _compute_filtered_named_modules(self):
        filtered_named_modules = []
        for name, module in self.named_modules():
            # print(name, all([x not in name for x in ['Dropout','ReLU']]))
            if all([x not in name for x in ['Dropout', 'ReLU']]):
                filtered_named_modules += [(name, module)]
        filtered_named_modules = filtered_named_modules[1:-1]
        self.filtered_named_modules = filtered_named_modules

    def _setup_heuristics(self, cfg_heuristics):
        self.heuristics = {}
        if cfg_heuristics is None or len(cfg_heuristics) == 0:
            self.logger.info("No training heuristics.")
        else:
            self.logger.info(f"Train heuristics: {[x for x in cfg_heuristics]}")
            for cfg_name in cfg_heuristics:
                if cfg_name == 'bias_shaping':
                    self.heuristics[cfg_name] = BiasShaping(self, cfg_heuristics[cfg_name])
                elif cfg_name == 'rs_loss':
                    self.heuristics[cfg_name] = RSLoss(self, cfg_heuristics[cfg_name])
                elif cfg_name == 'prune':
                    self.heuristics[cfg_name] = Prune(self, cfg_heuristics[cfg_name])
                else:
                    raise NotImplementedError

    def run_heuristics(self, name, data):
        return self.heuristics[name].run(data)

    def register_activation_hocks(self):
        activation = {}
        for name, module in self.filtered_named_modules:
            module.register_forward_hook(self.get_activation(name, activation))
        return activation

    @staticmethod
    def get_activation(name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # estimate ReLUs

    def estimate_stable_ReLU(self, ReLU_est, test_loader=None):
        if ReLU_est == 'TD':
            safe_le_zero_all = []
            safe_ge_zero_all = []
            for layer in self.activation.keys():
                val = self.activation[layer].view(self.activation[layer].size(0), -1).cpu().numpy()
                val_min = np.min(val, axis=0)
                val_max = np.max(val, axis=0)
                safe_ge_zero = (np.asarray(val_min) >= 0).sum()
                safe_le_zero = (np.asarray(val_max) <= 0).sum()

                safe_le_zero_all += [safe_le_zero]
                safe_ge_zero_all += [safe_ge_zero]
            total_safe = sum(safe_le_zero_all)+sum(safe_ge_zero_all)

        elif ReLU_est == 'VS':
            eps = 0.2
            total_safe = []
            self.eval()
            with torch.no_grad():
                data, _ = next(iter(test_loader))
                data = data.to(self.device)
                adv = torch.FloatTensor(data.shape).uniform_(-eps, +eps).to(self.device)
                data = data+adv
                self(data)
                total_safe += [self.estimate_stable_ReLU('TD')]
            # safe = np.mean(np.array(safe))
            self.train()
            total_safe = np.mean(np.array(total_safe))
        else:
            raise NotImplementedError

        return total_safe
