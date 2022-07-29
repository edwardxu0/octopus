import numpy as np
import torch

from . import ReLUEstimator

# Naive Interval Propagation estimator
class NIPEstimator(ReLUEstimator):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.__name__ = "NIP ReLU Estimator"
        self.epsilon = kwargs["epsilon"]

    def propagate(self, **kwargs):
        data = kwargs["data"]
        data = data.view((-1, np.prod(self.model.artifact.input_shape)))

        lb = torch.maximum(data - self.epsilon, torch.tensor(0.0, requires_grad=True))
        ub = torch.minimum(data + self.epsilon, torch.tensor(1.0, requires_grad=True))

        lb_ = []
        ub_ = []
        le_0_ = []
        ge_0_ = []
        for name in list(self.model.layers.keys())[:-1]:
            layer = self.model.layers[name]
            if isinstance(layer, torch.nn.Linear):
                w = layer.weight
                b = layer.bias

                lb, ub = self._interval_arithmetic(lb, ub, w, b)
                # lb_ += [lb.view([1, *lb.shape])]
                # ub_ += [ub.view([1, *ub.shape])]
                lb_ += [lb]
                ub_ += [ub]

                le_0, ge_0 = ReLUEstimator._calculate_stable_ReLUs(lb, ub)
                # le_0_ += [le_0.view(1, *le_0.shape)]
                # ge_0_ += [ge_0.view(1, *ge_0.shape)]
                le_0_ += [le_0]
                ge_0_ += [ge_0]

            elif isinstance(layer, torch.nn.ReLU):
                lb = torch.relu(lb)
                ub = torch.relu(ub)

            elif isinstance(layer, torch.nn.Dropout):
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

    @staticmethod
    def _interval_arithmetic(lb, ub, W, b):
        W_max = torch.maximum(W, torch.tensor(0.0, requires_grad=True)).T
        W_min = torch.minimum(W, torch.tensor(0.0, requires_grad=True)).T
        new_lb = torch.matmul(lb, W_max) + torch.matmul(ub, W_min) + b
        new_ub = torch.matmul(ub, W_max) + torch.matmul(lb, W_min) + b
        return new_lb, new_ub
