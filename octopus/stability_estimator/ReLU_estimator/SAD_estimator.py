import torch

from . import ReLUEstimator
from .SDD_estimator import SDDEstimator


# Sampled Adversarial Estimator
class SADEstimator(ReLUEstimator):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.epsilon = kwargs["epsilon"]
        self.samples = kwargs["samples"]
        self.SDDEstimator = SDDEstimator(model)

    def run(self, **kwargs):
        test_loader = kwargs["test_loader"]

        data, _ = next(iter(test_loader))
        data = data[: self.samples].to(self.model.device)
        adv = (
            torch.FloatTensor(data.shape)
            .uniform_(-self.epsilon, +self.epsilon)
            .to(self.model.device)
        )
        data = data + adv

        self.model.eval()
        with torch.no_grad():
            self.model(data)

        le_0, ge_0, lb_, ub_ = self.SDDEstimator.run()
        self.model.train()

        return le_0, ge_0, lb_, ub_
