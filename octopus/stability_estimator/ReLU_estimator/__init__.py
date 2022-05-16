import torch

# ReLU estimator
class ReLUEstimator:
    def __init__(self, model):
        self.model = model
        self.logger = model.logger
        self.device = self.model.device

    @staticmethod
    def _calculate_stable_ReLUs(lb, ub):
        le_0 = torch.sum(ub <= 0).int()
        ge_0 = torch.sum(lb >= 0).int()
        return le_0, ge_0
