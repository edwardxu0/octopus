import torch

from . import Heuristic
from ..stability_estimator.ReLU_estimator.SIP_estimator import SIPEstimator


class Prune(Heuristic):
    def __init__(self, model, cfg):
        super().__init__(model, cfg["stable_estimator"])
        self.__name__ = "Prune"
        self.model = model
        self.mode = cfg["mode"]
        # self.pace = cfg["pace"]
        self.sparsity = cfg["sparsity"]
        self.re_arch = None if "re_arch" not in cfg else cfg["re_arch"]

        self._init_mask()

    # pruning code ...
    def run(self, **kwargs):
        re_arch_on = False
        # if kwargs["batch"] != 0 and kwargs["batch"] % self.pace == 0:
        if kwargs["batch_idx"] + 1 == kwargs["total_batches"]:
            # if True:
            self.logger.info("Prune starts here ...")
            # prune weights

            if self.mode == "structure":
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
                    self._restructure(self.mask)
        return re_arch_on

    # initialize the mask
    def _init_mask(self):
        self.mask = []
        for name, module in self.model.filtered_named_modules:
            if "FC" in name:
                self.mask += [torch.ones(module.out_features).to(self.model.device)]
            else:
                raise NotImplementedError

    def _update_mask(self, pr_ratio, **kwargs):
        self.stable_estimator.propagate(**kwargs)
        lb_, ub_ = self.stable_estimator.get_bounds()

        mean = []
        for lb, ub in zip(lb_, ub_):
            assert len(lb.shape) == 2
            m = torch.mean((abs(lb) + abs(ub)) / 2, axis=0)
            mean += [m]

        weight_sorted = torch.sort(torch.cat(mean).reshape(-1))[0]
        weight_sorted_nonzero = weight_sorted[weight_sorted.nonzero().squeeze()]

        nb_neurons = sum([sum(x) for x in self.mask])
        threshold = weight_sorted_nonzero[int(pr_ratio * nb_neurons)]

        for i, m in enumerate(mean):
            self.mask[i][m < threshold] = 0
            if torch.where(self.mask[i] == 1)[0].nelement() == 0:
                raise ValueError("Pruning an entire layer is bad idea.")

    def _apply_mask(self):
        for i, (_, layer) in enumerate(self.model.filtered_named_modules):
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data *= self.mask[i].unsqueeze(1)
            elif isinstance(layer, torch.nn.Conv2d):
                assert False
                layer.weight.data *= self.mask[i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            else:
                assert False

    # restructure architecture to remove empty neurons
    def _restructure(self, mask):
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
        if isinstance(self.stable_estimator, SIPEstimator):
            self.stable_estimator.init_inet()

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
