import numpy as np
import torch


from . import ReLUEstimator

# Naive Interval Propagation estimator
class NIPEstimator(ReLUEstimator):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.epsilon = kwargs["epsilon"]

    def run(self, **kwargs):
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

                lb_batch = torch.min(lb, axis=0).values
                ub_batch = torch.max(ub, axis=0).values
                lb_ += [lb_batch]
                ub_ += [ub_batch]
                # print(lb_batch)
                # print(ub_batch)
                le_0, ge_0 = ReLUEstimator._calculate_stable_ReLUs(lb_batch, ub_batch)
                le_0_ += [le_0.view(1)]
                ge_0_ += [ge_0.view(1)]

            elif isinstance(layer, torch.nn.ReLU):
                lb = torch.relu(lb)
                ub = torch.relu(ub)

            elif isinstance(layer, torch.nn.Dropout):
                pass

            else:
                raise NotImplementedError(layer)

        le_0_ = torch.cat(le_0_, dim=0)
        ge_0_ = torch.cat(ge_0_, dim=0)
        return le_0_, ge_0_, lb_, ub_

    @staticmethod
    def _interval_arithmetic(lb, ub, W, b):
        W_max = torch.maximum(W, torch.tensor(0.0, requires_grad=True)).T
        W_min = torch.minimum(W, torch.tensor(0.0, requires_grad=True)).T
        new_lb = torch.matmul(lb, W_max) + torch.matmul(ub, W_min) + b
        new_ub = torch.matmul(ub, W_max) + torch.matmul(lb, W_min) + b
        return new_lb, new_ub
