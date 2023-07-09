import torch
from torch.nn import Linear, Conv2d
import numpy as np

from . import ReLUEstimator


# Sampled Distribution Estimator
class SDDEstimator(ReLUEstimator):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.__name__ = "SDD ReLU Estimator"

    def propagate(self, **kwargs):
        lb_ = []
        ub_ = []
        le_0_ = []
        ge_0_ = []
        raw_ = []
        for name, layer in self.model.filtered_named_modules:
            X = self.model._batch_values[name]
            raw_ += [X]

            # get min of kernels
            if isinstance(layer, Linear):
                lb = torch.min(X, axis=0).values
                ub = torch.max(X, axis=0).values

            elif isinstance(layer, Conv2d):
                X = torch.flatten(X, start_dim=2)
                X_min = torch.min(X, axis=-1).values
                X_max = torch.max(X, axis=-1).values
                lb = torch.min(X_min, axis=0).values
                ub = torch.max(X_max, axis=0).values

            else:
                # raise NotImplementedError(layer)
                pass

            lb = lb.view(1, *lb.shape)
            ub = ub.view(1, *ub.shape)

            # lb_ += [lb.view(1, *lb.shape)]
            # ub_ += [ub.view(1, *lb.shape)]
            lb_ += [lb]
            ub_ += [ub]
            # print("SDD lb", lb.shape)
            le_0, ge_0 = ReLUEstimator._calculate_stable_ReLUs(lb, ub)
            # print("SDD le_0", le_0.shape)
            # le_0_ += [le_0.view(1, *le_0.shape)]
            # ge_0_ += [ge_0.view(1, *ge_0.shape)]
            le_0_ += [le_0]
            ge_0_ += [ge_0]

        # self.stable_le_0_ = torch.cat(le_0_, dim=0)
        # self.stable_ge_0_ = torch.cat(ge_0_, dim=0)
        self.stable_le_0_ = le_0_
        self.stable_ge_0_ = ge_0_
        # self.lb_ = torch.cat(lb_, dim=0)
        # self.ub_ = torch.cat(ub_, dim=0)
        self.lb_ = lb_
        self.ub_ = ub_
        self.raw_ = raw_

        # for i in range(len(self.lb_)):
        #    print(self.lb_[i].shape)
        #    print(self.ub_[i].shape)
        # exit()
