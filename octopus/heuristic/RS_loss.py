import torch

from . import Heuristic


class RSLoss(Heuristic):
    def __init__(self, model, cfg):
        super().__init__(model, cfg["stable_estimator"])
        self.__name__ = "RS Loss"
        self.mode = cfg["mode"]
        assert cfg["mode"] == "standard"

    def run(self, **kwargs):
        self.stable_estimator.propagate(**kwargs)
        lb_, ub_ = self.stable_estimator.get_bounds()
        loss = []
        for lb, ub in zip(lb_, ub_):
            assert len(lb.shape) == 2
            rs_loss = self._l_relu_stable(lb, ub)
            loss += [rs_loss.view(1)]

        loss = torch.cat(loss)
        loss = torch.sum(loss)
        return loss

    # RS Loss Function
    def _l_relu_stable(self, lb, ub, norm_constant=1.0):
        loss = torch.mean(
            -torch.sum(
                torch.tanh(
                    torch.tensor(1.0, requires_grad=True) + norm_constant * lb * ub
                ),
                axis=-1,
            )
        )
        return loss
