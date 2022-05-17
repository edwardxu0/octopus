import torch
import numpy as np

from . import ReLUEstimator


# Sampled Distribution Estimator
class SDDEstimator(ReLUEstimator):
    def __init__(self, model, **kwargs):
        super().__init__(model)

    def propagate(self, **kwargs):
        lb_ = []
        ub_ = []
        le_0_ = []
        ge_0_ = []
        for name, _ in self.model.filtered_named_modules:
            X = self.model._batch_values[name]
            lb = torch.min(X, axis=0).values
            ub = torch.max(X, axis=0).values
            lb = lb.view(1, *lb.shape)
            ub = ub.view(1, *ub.shape)
            lb_ += [lb.view(1, *lb.shape)]
            ub_ += [ub.view(1, *lb.shape)]

            le_0, ge_0 = ReLUEstimator._calculate_stable_ReLUs(lb, ub)
            le_0_ += [le_0.view(1, *ge_0.shape)]
            ge_0_ += [ge_0.view(1, *ge_0.shape)]

        self.stable_le_0_ = torch.cat(le_0_, dim=0)
        self.stable_ge_0_ = torch.cat(ge_0_, dim=0)
        self.lb_ = torch.cat(lb_, dim=0)
        self.ub_ = torch.cat(ub_, dim=0)
