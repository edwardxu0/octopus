import numpy as np
import torch
import torch.nn as nn

from . import ReLUEstimator

from symbolic_interval.symbolic_network import Interval_network
from symbolic_interval.interval import Symbolic_interval

# Naive Interval Propagation estimator
class SIPEstimator(ReLUEstimator):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.__name__ = "SIP ReLU Estimator"
        self.epsilon = kwargs["epsilon"]
        self.init_inet()

    def init_inet(self):
        layers = list(self.model.layers.values())
        self.layers = [x for x in layers if not isinstance(x, nn.Dropout)]
        inet_model = nn.Sequential(*layers).to(self.model.device)
        self.inet = Interval_network(inet_model, None)

    def propagate(self, **kwargs):
        data = kwargs["data"]
        X = data.view((-1, np.prod(self.model.artifact.input_shape)))

        minimum = X.min().item()
        maximum = X.max().item()
        ix = Symbolic_interval(
            torch.clamp(X - self.epsilon, minimum, maximum),
            torch.clamp(X + self.epsilon, minimum, maximum),
            use_cuda=self.model.device == torch.device("cuda"),
        )
        ixo = self.inet(ix)

        lb_ = []
        ub_ = []
        le_0_ = []
        ge_0_ = []
        assert len(self.inet.intermediate_ix) == len(self.layers)
        for i, l in enumerate(self.inet.intermediate_ix[:-1]):
            layer = self.layers[i]
            if isinstance(layer, torch.nn.Linear):
                # lb_ += [l.l.view([1, *l.l.shape])]
                # ub_ += [l.u.view([1, *l.u.shape])]
                lb_ += [l.l]
                ub_ += [l.u]
                le_0, ge_0 = ReLUEstimator._calculate_stable_ReLUs(l.l, l.u)
                # le_0_ += [le_0.view(1, *le_0.shape)]
                # ge_0_ += [ge_0.view(1, *ge_0.shape)]
                le_0_ += [le_0]
                ge_0_ += [ge_0]

            elif isinstance(layer, torch.nn.ReLU):
                pass
            else:
                raise NotImplementedError(layer)

        # self.stable_le_0_ = torch.cat(le_0_, dim=0)
        # self.stable_ge_0_ = torch.cat(ge_0_, dim=0)
        self.stable_le_0_ = le_0_
        self.stable_ge_0_ = ge_0_
        # self.lb_ = torch.cat(lb_, dim=0)
        # self.ub_ = torch.cat(ub_, dim=0)
        self.lb_ = lb_
        self.ub_ = ub_

        del ix
        del ixo
        del self.inet.intermediate_ix
        self.inet.intermediate_ix = []
