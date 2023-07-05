import torch
from torch.nn import Linear, Conv2d, ReLU

from . import Stabilizer
from ..stability_estimator.ReLU_estimator.SIP_estimator import SIPEstimator


class StablePrune(Stabilizer):
    def __init__(self, model, cfg):
        super().__init__(model, cfg["stable_estimators"])
        self.__name__ = "Prune"
        self.model = model
        self.mode = cfg["mode"]
        self.pace = cfg["pace"]
        self.sparsity = cfg["sparsity"]
        self.re_arch = None if "re_arch" not in cfg else cfg["re_arch"]

        self._init_mask()

    def run(self, **kwargs):
        re_arch_on = False
        if (kwargs["batch_idx"] + 1) % self.pace == 0:
            # if True:
            self.logger.info("Prune starts here ...")
            # prune weights

            if self.mode in ["dropnet", "stablenet"]:
                self.logger.info("Using iterative structure pruning ...")
                self._update_mask(self.sparsity, **kwargs)
                self._apply_mask()
            else:
                raise NotImplementedError(f"Prune mode: {self.mode} is not supported.")

            # restructure network
            if self.re_arch:
                if self.re_arch == "all":
                    re_arch_on = True
                elif self.re_arch == "last":
                    if kwargs["epoch"] == kwargs["total_epoch"]:
                        self.logger.warning(
                            "re_arch on last EPOCH, not last pruning instance."
                        )
                        re_arch_on = True

                else:
                    raise ValueError(self.re_arch)

                if re_arch_on:
                    self.logger.info("Restructuring network ...")
                    self._restructure()
        return re_arch_on

    # initialize the mask
    def _init_mask(self):
        self.mask = []
        for name, module in self.model.filtered_named_modules:
            if isinstance(module, Linear):
                self.mask += [torch.ones(module.out_features).to(self.model.device)]
            elif isinstance(module, Conv2d):
                self.mask += [torch.ones(module.out_channels).to(self.model.device)]
            else:
                raise NotImplementedError

    def _update_mask(self, pr_ratio, **kwargs):
        self.stable_estimators.propagate(**kwargs)

        if self.mode == "dropnet":
            lb_, ub_ = self.stable_estimators.get_bounds()
            mean = []
            for lb, ub in zip(lb_, ub_):
                assert len(lb.shape) == 2
                m = torch.mean((abs(lb) + abs(ub)) / 2, axis=0)
                mean += [m]

            weight_sorted = torch.sort(torch.cat(mean).reshape(-1))[0]
            weight_sorted_nonzero = weight_sorted[weight_sorted.nonzero().squeeze()]

            nb_neurons = sum([sum(x) for x in self.mask])
            threshold = weight_sorted_nonzero[int(pr_ratio * nb_neurons)]

        elif self.mode == "stablenet":
            lb_, ub_ = self.stable_estimators.get_bounds()
            mean = []
            # check conv layers
            for i, (lb, ub) in enumerate(zip(lb_, ub_)):
                if len(ub.shape) == 4:
                    ub = torch.amax(ub, dim=(2, 3))

                m = torch.mean(ub, axis=0)

                mean += [m]
            values = torch.sort(torch.cat(mean).reshape(-1)).values

            nb_neurons = len(values)
            nb_unstable = len(values[values > 0])
            nb_stable = nb_neurons - nb_unstable

            threshold = values[nb_stable - 1 + int(pr_ratio * nb_unstable)]

        else:
            assert False

        for i, m in enumerate(mean):
            self.mask[i][m < threshold] = 0
        # if torch.where(self.mask[i] == 1)[0].nelement() == 0:
        #    raise ValueError("Pruning an entire layer is bad idea.")

    def _apply_mask(self):
        for i, (_, layer) in enumerate(self.model.filtered_named_modules):
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data *= self.mask[i].unsqueeze(1)
            elif isinstance(layer, torch.nn.Conv2d):
                for j, x in enumerate(layer.weight.data):
                    x *= self.mask[i][j]
            else:
                assert False

    # restructure architecture to remove empty neurons
    def _restructure(self):
        # compute new weights/bias
        weight_, bias_ = self._filter_parameters()
        # self.logger.debug(f"Remaining # neurons: {sum([b.shape[0] for b in bias])}")
        layers = [x.shape[0] for x in bias_][:-1]
        # clear model
        self.model.clear()
        # construct new model

        self.model.set_layers(layers, weight_, bias_)
        self.logger.info(f"Restructured model: \n{self.model}")
        self.logger.info(f"# ReLUs: {self.model.nb_ReLUs}")

        self._init_mask()
        if isinstance(self.stable_estimators, SIPEstimator):
            self.stable_estimators.init_inet()

    def _filter_parameters(self):
        weight_ = []
        bias_ = []
        layers = [x[1] for x in self.model.filtered_named_modules] + [
            list(self.model.layers.values())[-1]
        ]
        for i, layer in enumerate(layers):
            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight.data
                bias = layer.bias.data
                if i + 1 != len(layers):
                    weight = weight[torch.where(self.mask[i] == 1)]
                    bias = bias[torch.where(self.mask[i] == 1)]
                if i != 0:
                    weight = weight.T[torch.where(self.mask[i - 1] == 1)].T
                weight_ += [weight]
                bias_ += [bias]
        return weight_, bias_
