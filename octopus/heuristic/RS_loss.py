import torch
import numpy as np

from . import Heuristic


class RSLoss(Heuristic):
    def __init__(self, model, cfg):
        super().__init__(model.logger)
        self.model = model
        self.mode = cfg["mode"]
        assert cfg["mode"] == "standard"
        self.epsilon = cfg["epsilon"]

    def run(self, **kwargs):
        data = kwargs["data"]
        loss = torch.zeros((len(data), 3), device=self.model.device)
        data = data.view((-1, np.prod(self.model.artifact.input_shape)))
        lb = torch.maximum(data - self.epsilon, torch.tensor(0.0, requires_grad=True))
        ub = torch.minimum(data + self.epsilon, torch.tensor(1.0, requires_grad=True))

        for i, (name, _) in enumerate(self.model.filtered_named_modules):
            w = self.model.__getattr__(name).weight
            b = self.model.__getattr__(name).bias
            lb, ub = self._interval_arithmetic(lb, ub, w, b)
            rs_loss = self._l_relu_stable(lb, ub)
            loss[i] += rs_loss
        loss = torch.sum(torch.mean(loss, axis=1))
        return loss

    """RS Loss Function"""

    def _l_relu_stable(self, lb, ub, norm_constant=1.0):
        loss = -torch.sum(
            torch.tanh(torch.tensor(1.0, requires_grad=True) + norm_constant * lb * ub)
        )
        return loss

    @staticmethod
    def _interval_arithmetic(lb, ub, W, b):
        W_max = torch.maximum(W, torch.tensor(0.0, requires_grad=True)).T
        W_min = torch.minimum(W, torch.tensor(0.0, requires_grad=True)).T
        new_lb = torch.matmul(lb, W_max) + torch.matmul(ub, W_min) + b
        new_ub = torch.matmul(ub, W_max) + torch.matmul(lb, W_min) + b
        return new_lb, new_ub

    # TODO: refine this ...
    # sparsity of weight matrix
    # unused

    def get_WD_loss(self, device):
        assert False
        loss = torch.zeros(3).to(device=device)
        params = [14 * 14, 7 * 7, 1]
        c = 0
        for layer in self.state_dict():
            if "weight" in layer and "out" not in layer:
                w = self.state_dict()[layer]
                x = torch.abs(w)
                x = torch.sum(w) * params[c]
                loss[c] = x
                c += 1
        loss = torch.sum(loss)
        return loss
