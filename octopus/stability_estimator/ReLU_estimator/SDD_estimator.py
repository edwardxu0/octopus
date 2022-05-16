import torch
import numpy as np

from . import ReLUEstimator


# Sampled Distribution Estimator
class SDDEstimator(ReLUEstimator):
    def __init__(self, model, **kwargs):
        super().__init__(model)

    def run(self, **kwargs):
        lb_ = []
        ub_ = []
        le_0_ = []
        ge_0_ = []

        for name, _ in self.model.filtered_named_modules:
            X = self.model._batch_values[name]
            lb = torch.min(X, axis=0).values
            ub = torch.max(X, axis=0).values
            lb_ += [lb]
            ub_ += [ub]

            le_0, ge_0 = ReLUEstimator._calculate_stable_ReLUs(lb, ub)
            le_0_ += [le_0.view(1)]
            ge_0_ += [ge_0.view(1)]

        le_0_ = torch.cat(le_0_, dim=0)
        ge_0_ = torch.cat(ge_0_, dim=0)

        return le_0_, ge_0_, lb_, ub_
