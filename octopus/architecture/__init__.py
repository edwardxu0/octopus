from torch import nn
from torch.nn import Linear, Conv2d, ReLU
import numpy as np


from ..heuristic.prune import Prune
from ..heuristic.RS_loss import RSLoss
from ..heuristic.bias_shaping import BiasShaping


class BasicNet(nn.Module):
    def __init__(self, logger, device, amp):
        super(BasicNet, self).__init__()
        self.logger = logger
        self.device = device
        self.amp = amp

    def __setup__(self):
        # self._compute_filtered_named_modules()
        # self.activation = self.register_activation_hocks()

        filtered_named_modules = []
        # for i, name, module in self.named_modules():
        for i in range(len(self.layers) - 1):
            x = list(self.layers.keys())[i]
            x_ = list(self.layers.keys())[i + 1]
            l = self.layers[x]
            l_ = self.layers[x_]
            if any([isinstance(l, y) for y in [Linear, Conv2d]]) and isinstance(
                l_, ReLU
            ):
                filtered_named_modules += [(x, l)]
        self.filtered_named_modules = filtered_named_modules

    def _compute_filtered_named_modules(self):
        filtered_named_modules = []
        for name, module in self.named_modules():
            # print(name, all([x not in name for x in ['Dropout','ReLU']]))
            if all([x not in name for x in ["Dropout", "ReLU"]]):
                filtered_named_modules += [(name, module)]
        filtered_named_modules = filtered_named_modules[1:-1]
        self.filtered_named_modules = filtered_named_modules

    def _setup_heuristics(self, cfg_heuristics):
        self.heuristics = {}
        if cfg_heuristics is None or len(cfg_heuristics) == 0:
            self.logger.info("No training heuristics.")
        else:

            for cfg_name in cfg_heuristics:
                if cfg_name == "bias_shaping":
                    self.heuristics[cfg_name] = BiasShaping(
                        self, cfg_heuristics[cfg_name]
                    )
                elif cfg_name == "rs_loss":
                    self.heuristics[cfg_name] = RSLoss(self, cfg_heuristics[cfg_name])
                elif cfg_name == "prune":
                    self.heuristics[cfg_name] = Prune(self, cfg_heuristics[cfg_name])
                else:
                    raise NotImplementedError
            self.logger.info(
                f"Train heuristics: {[f'{self.heuristics[x].__name__}({self.heuristics[x].stable_estimator.__name__.split()[0]})' for x in self.heuristics]}"
            )

    def run_heuristics(self, name, **kwargs):
        return self.heuristics[name].run(**kwargs)

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
