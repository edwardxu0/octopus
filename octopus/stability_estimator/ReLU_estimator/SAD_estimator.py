import numpy as np
import torch

from . import ReLUEstimator
from .SDD_estimator import SDDEstimator


class SADEstimator(ReLUEstimator):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.epsilon = kwargs["epsilon"]
        self.samples = kwargs["samples"]
        self.SDDEstimator = SDDEstimator(model)

    def run(self, **kwargs):
        test_loader = kwargs["test_loader"]
        device = kwargs["device"]

        total_safe = []
        self.model.eval()
        with torch.no_grad():
            data, _ = next(iter(test_loader))
            data = data.to(device)
            adv = (
                torch.FloatTensor(data.shape)
                .uniform_(-self.epsilon, +self.epsilon)
                .to(device)
            )
            data = data + adv
            self.model(data)
            total_safe += [self.SDDEstimator.run()]
        # safe = np.mean(np.array(safe))
        self.model.train()
        total_safe = np.mean(np.array(total_safe))

        return total_safe
