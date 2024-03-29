import torch

# ReLU estimator
class ReLUEstimator:
    def __init__(self, model):
        self.model = model
        self.logger = model.logger
        self.device = self.model.device

    def get_stable_ReLUs(self):
        if any(x not in self.__dict__ for x in ["stable_le_0_", "stable_ge_0_"]):
            raise ReferenceError(f'Did you run "{self}".propagate()?')
        return self.stable_le_0_, self.stable_ge_0_

    def get_bounds(self):
        if any(x not in self.__dict__ for x in ["lb_", "ub_"]):
            raise ReferenceError(f'Did you run "{self}".propagate()?')
        return self.lb_, self.ub_

    def get_raw(self):
        if any(x not in self.__dict__ for x in ["raw_"]):
            raise ReferenceError(f'Did you run "{self}".propagate()?')
        return self.raw_

    def clean(self):
        pass

    @staticmethod
    def _calculate_stable_ReLUs(lb, ub):
        if len(lb.shape) == 4 and len(ub.shape) == 4:
            lb = lb.flatten(start_dim=1)
            ub = ub.flatten(start_dim=1)

        le_0 = torch.sum(ub < 0, axis=-1).int()
        ge_0 = torch.sum(lb >= 0, axis=-1).int()

        return le_0, ge_0
