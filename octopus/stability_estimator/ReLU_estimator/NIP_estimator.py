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
        # data = data.view(-1, np.prod(data.shape[1:]))
        # data = data.view((-1, np.prod(self.model.artifact.input_shape)))
        lb = torch.maximum(data - self.epsilon, torch.tensor(0.0, requires_grad=True))
        ub = torch.minimum(data + self.epsilon, torch.tensor(1.0, requires_grad=True))

        lb_ = []
        ub_ = []
        le_0_ = []
        ge_0_ = []
        for name in list(self.model.layers.keys())[:-1]:
            layer = self.model.layers[name]
            if isinstance(layer, torch.nn.Linear):
                lb = lb.view(-1, np.prod(lb.shape[1:]))
                ub = ub.view(-1, np.prod(ub.shape[1:]))
                w = layer.weight
                b = layer.bias

                # print("start fc")
                # print(lb.shape, w.shape, b.shape)

                lb, ub = self._interval_arithmetic(lb, ub, w, b)
                # lb_ += [lb.view([1, *lb.shape])]
                # ub_ += [ub.view([1, *ub.shape])]
                lb_ += [lb]
                ub_ += [ub]
                # print("lb.shape2", lb.shape)

                le_0, ge_0 = ReLUEstimator._calculate_stable_ReLUs(lb, ub)
                # le_0_ += [le_0.view(1, *le_0.shape)]
                # ge_0_ += [ge_0.view(1, *ge_0.shape)]
                le_0_ += [le_0]
                ge_0_ += [ge_0]

                # print("end fc")

            elif isinstance(layer, torch.nn.Conv2d):
                w = layer.weight
                b = layer.bias
                s = layer.stride
                p = layer.padding

                # print("start")
                # print(lb.shape, w.shape, b.shape)

                # print(lb.shape)
                lb, ub = self._interval_arithmetic_conv_2x2(lb, ub, w, b, s, p)
                # print(lb.shape, ub.shape)

                # lb_ += [lb.view([1, *lb.shape])]
                # ub_ += [ub.view([1, *ub.shape])]

                lb_ += [lb]
                ub_ += [ub]

                # print(lb.flatten().shape)

                le_0, ge_0 = ReLUEstimator._calculate_stable_ReLUs(lb, ub)
                # le_0_ += [le_0.view(1, *le_0.shape)]
                # ge_0_ += [ge_0.view(1, *ge_0.shape)]
                le_0_ += [le_0]
                ge_0_ += [ge_0]

                # print("done conv")

            elif isinstance(layer, torch.nn.ReLU):
                # print("ReLU")
                lb = torch.relu(lb)
                ub = torch.relu(ub)

            elif isinstance(layer, torch.nn.Dropout):
                continue

            elif isinstance(layer, torch.nn.Flatten):
                continue

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

    @staticmethod
    def _interval_arithmetic_conv_2x2(lb, ub, W, b, s, p):
        W_max = torch.maximum(W, torch.tensor(0.0, requires_grad=True))
        W_min = torch.minimum(W, torch.tensor(0.0, requires_grad=True))

        new_lb = NIPEstimator._conv2d_2x2(
            lb, W_max, b, s, p
        ) + NIPEstimator._conv2d_2x2(ub, W_min, b, s, p)

        new_ub = NIPEstimator._conv2d_2x2(
            ub, W_max, b, s, p
        ) + NIPEstimator._conv2d_2x2(lb, W_min, b, s, p)

        return new_lb, new_ub

    @staticmethod
    def _conv2d_2x2(x, W, b, s, p):
        return torch.nn.functional.conv2d(x, W, b, stride=s, padding=p)
