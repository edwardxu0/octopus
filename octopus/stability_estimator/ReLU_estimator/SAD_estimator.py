import torch

from . import ReLUEstimator
from .SDD_estimator import SDDEstimator


# Sampled Adversarial Estimator
class SADEstimator(ReLUEstimator):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.__name__ = 'SAD ReLU Estimator'
        self.epsilon = kwargs["epsilon"]
        self.samples = kwargs["samples"]
        self.SDDEstimator = SDDEstimator(model)

    def propagate(self, **kwargs):
        test_loader = kwargs["test_loader"]

        data, _ = next(iter(test_loader))
        data = data[: self.samples].to(self.model.device)
        adv = (
            torch.FloatTensor(data.shape)
            .uniform_(-self.epsilon, +self.epsilon)
            .to(self.model.device)
        )
        data = data + adv

        # clipping data to 0 and 1
        data = torch.clamp(data, min=0, max=1)

        self.model.eval()
        with torch.no_grad():
            self.model(data)
        self.SDDEstimator.propagate(**kwargs)
        self.model.train()

        self.stable_le_0_ = self.SDDEstimator.stable_le_0_
        self.stable_ge_0_ = self.SDDEstimator.stable_ge_0_
        self.lb_ = self.SDDEstimator.lb_
        self.ub_ = self.SDDEstimator.ub_
