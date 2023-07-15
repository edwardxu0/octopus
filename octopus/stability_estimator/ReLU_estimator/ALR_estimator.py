import numpy as np
import torch
import copy
from torch.nn import Linear, Conv2d

from . import ReLUEstimator

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import warnings

warnings.filterwarnings("ignore", module="auto_LiRPA")
# warnings.filterwarnings("ignore", module="torch")


# Naive Interval Propagation estimator
class ALREstimator(ReLUEstimator):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.__name__ = "ALR ReLU Estimator"
        self.epsilon = kwargs["epsilon"]
        self.model = model
        self.device = model.device
        self.method = kwargs["method"]
        self.bounded_model = BoundedModule(
            copy.deepcopy(self.model),
            torch.zeros([1] + self.model.artifact.input_shape).to(self.device),
            bound_opts={"conv_mode": "patches"},
            device=self.device,
        )

    def get_hidden_bounds(self):
        lower_bounds = [
            layer.inputs[0].lower.detach()
            for layer in self.bounded_model.perturbed_optimizable_activations
        ]
        upper_bounds = [
            layer.inputs[0].upper.detach()
            for layer in self.bounded_model.perturbed_optimizable_activations
        ]
        return lower_bounds, upper_bounds

    def propagate(self, **kwargs):
        self.model.eval()
        self.bounded_model = BoundedModule(
            self.model.clone(),
            torch.zeros([1] + self.model.artifact.input_shape).to(self.device),
            bound_opts={"conv_mode": "patches"},
            device=self.device,
        )
        data = kwargs["data"]
        X = data.view((-1, np.prod(self.model.artifact.input_shape)))
        # X = data

        ptb = PerturbationLpNorm(norm=np.inf, eps=self.epsilon)
        # Input tensor is wrapped in a BoundedTensor object.
        bounded_image = BoundedTensor(X, ptb).to(self.device)
        with torch.no_grad():  # If gradients of the bounds are not needed, we can use no_grad to save memory.
            self.bounded_model.eval()
            lb, ub = self.bounded_model.compute_bounds(
                x=(bounded_image,), method=self.method
            )
            self.bounded_model.train()

        lb_, ub_ = self.get_hidden_bounds()
        le_0_ = []
        ge_0_ = []
        for i, (name, layer) in enumerate(self.model.filtered_named_modules):
            # get min of kernels
            if isinstance(layer, Linear):
                lb = lb_[i]
                ub = ub_[i]

            elif isinstance(layer, Conv2d):
                raise NotImplementedError(layer)
            else:
                raise NotImplementedError(layer)

            # lb = lb.view(1, *lb.shape)
            # ub = ub.view(1, *ub.shape)

            # lb_ += [lb.view(1, *lb.shape)]
            # ub_ += [ub.view(1, *lb.shape)]

            # print("ALR lb", lb.shape)
            le_0, ge_0 = ReLUEstimator._calculate_stable_ReLUs(lb, ub)
            # le_0_ += [le_0.view(1, *le_0.shape)]
            # ge_0_ += [ge_0.view(1, *ge_0.shape)]
            le_0_ += [le_0]
            ge_0_ += [ge_0]
            # print("ALR le_0", le_0.shape)

        # s = 0
        # for x in le_0_:
        #    s += sum(x)
        # for x in ge_0_:
        #    s += sum(x)
        # print("total: ", s)

        # self.stable_le_0_ = torch.cat(le_0_, dim=0)
        # self.stable_ge_0_ = torch.cat(ge_0_, dim=0)
        self.stable_le_0_ = le_0_
        self.stable_ge_0_ = ge_0_
        # self.lb_ = torch.cat(lb_, dim=0)
        # self.ub_ = torch.cat(ub_, dim=0)
        self.lb_ = lb_
        self.ub_ = ub_
