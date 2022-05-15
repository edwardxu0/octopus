import numpy as np

from . import ReLUEstimator

# Training distribution estimator
class SDDEstimator(ReLUEstimator):
    def __init__(self, model, **kwargs):
        super().__init__(model)

    def run(self):
        safe_le_zero_all = []
        safe_ge_zero_all = []
        for layer in self.model.activation.keys():
            val = (
                self.model.activation[layer]
                .view(self.model.activation[layer].size(0), -1)
                .cpu()
                .numpy()
            )
            val_min = np.min(val, axis=0)
            val_max = np.max(val, axis=0)
            safe_ge_zero = (np.asarray(val_min) >= 0).sum()
            safe_le_zero = (np.asarray(val_max) <= 0).sum()

            safe_le_zero_all += [safe_le_zero]
            safe_ge_zero_all += [safe_ge_zero]
        total_safe = sum(safe_le_zero_all) + sum(safe_ge_zero_all)
        return total_safe
