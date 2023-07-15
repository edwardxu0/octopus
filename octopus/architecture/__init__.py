import copy
from torch import nn
from torch.nn import Linear, Conv2d, ReLU


from ..stabilizer.stable_prune import StablePrune
from ..stabilizer.RS_loss import RSLoss
from ..stabilizer.bias_shaping import BiasShaping

# from .ReLUNet import ReLUNet


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

    def _setup_stabilizers(self, cfg_stabilizers):
        self.stabilizers = {}
        if cfg_stabilizers is None or len(cfg_stabilizers) == 0:
            self.logger.info("No training stabilizers.")
        else:
            for cfg_name in cfg_stabilizers:
                if cfg_name == "bias_shaping":
                    self.stabilizers[cfg_name] = BiasShaping(
                        self, cfg_stabilizers[cfg_name]
                    )
                elif cfg_name == "rs_loss":
                    self.stabilizers[cfg_name] = RSLoss(self, cfg_stabilizers[cfg_name])
                elif cfg_name == "stable_prune":
                    self.stabilizers[cfg_name] = StablePrune(
                        self, cfg_stabilizers[cfg_name]
                    )
                else:
                    raise NotImplementedError
            self.logger.info(
                f"Train stabilizers: {[f'{self.stabilizers[x].__name__}({self.stabilizers[x].stable_estimators.__name__.split()[0]})' for x in self.stabilizers]}"
            )

    def run_stabilizers(self, name, **kwargs):
        return self.stabilizers[name].run(**kwargs)

    def register_activation_hocks(self):
        activation = {}
        for name, module in self.filtered_named_modules:
            module.register_forward_hook(self.get_activation(name, activation))
        return activation

    def clone(self):
        from .ReLUNet import ReLUNet

        if isinstance(self, ReLUNet):
            new_model = ReLUNet(
                self.artifact, self.layers_configs, self.logger, self.device, self.amp
            )
        else:
            assert False

        pretrained = copy.deepcopy(self.state_dict())
        new_model.load_state_dict(pretrained)
        return new_model

    @staticmethod
    def get_activation(name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook
