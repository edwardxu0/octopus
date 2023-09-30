import torch
import torch.nn as nn
import numpy as np

from . import Stabilizer

from ..stability_estimator import get_stability_estimators
from ..stability_estimator.ReLU_estimator.NIP_estimator import NIPEstimator


class BiasShaping(Stabilizer):
    def __init__(self, model, cfg):
        super().__init__(model, cfg["stable_estimators"])
        self.__name__ = "Bias Shaping"

        self.mode = cfg["mode"]
        self.intensity = cfg["intensity"]
        self.occurrence = cfg["occurrence"] if "occurrence" in cfg else None
        self.pace = cfg["pace"] if "pace" in cfg else None
        self.decay = 1 if not "decay" in cfg else cfg["decay"]
        self.device = model.device
        assert not (self.occurrence and self.pace)

    def run(self, **kwargs):
        epoch = kwargs["epoch"]
        data = kwargs["data"]
        BS_switch = False

        # probability check
        if self.occurrence:
            BS_switch = np.random.rand() < self.occurrence * self.decay**epoch
        elif self.pace:
            BS_switch = kwargs["batch"] % self.pace == 0
        else:
            assert False
        gt = 0
        lt = 0
        if BS_switch:
            if self.mode == "all-layers":
                print("Executing all layer BS...")
                self.stable_estimators.propagate(data=data)
                val_min_lt_zeros = []
                val_max_gt_zeros = []

                for i, (name, layer) in enumerate(self.model.filtered_named_modules):
                    # print(name, layer)
                    if any(isinstance(layer, x) for x in [nn.Linear, nn.Conv2d]):
                        lb_, ub_ = self.stable_estimators.get_bounds()

                        if len(lb_[i].shape) == 2:
                            val_min = torch.min(lb_[i].T, axis=-1).values
                            val_max = torch.max(ub_[i].T, axis=-1).values
                        elif len(lb_[i].shape) == 4:
                            val_min = torch.amin(lb_[i], dim=(0, 2, 3))
                            val_max = torch.amax(ub_[i], dim=(0, 2, 3))
                        else:
                            raise ValueError(len(lb_[i].shape))

                        (
                            safe_le_zero,
                            safe_ge_zero,
                        ) = self.stable_estimators._calculate_stable_ReLUs(
                            val_min, val_max
                        )

                        val_min_lt_zero = np.copy(val_min.detach().cpu().numpy())
                        val_max_gt_zero = np.copy(val_max.detach().cpu().numpy())

                        val_min_lt_zeros += [val_min_lt_zero]
                        val_max_gt_zeros += [val_max_gt_zero]

                val_min_lt_zero = np.concatenate(val_min_lt_zeros)
                val_max_gt_zero = np.concatenate(val_max_gt_zeros)

                # pick lb < 0
                val_min_lt_zero[val_min_lt_zero >= 0] = 0
                # print(val_min_lt_zero)
                val_min_lt_zero *= -1
                # pick ub > 0
                val_max_gt_zero[val_max_gt_zero <= 0] = 0
                # print(val_max_gt_zero)

                val_abs_min = np.min(
                    np.array([val_min_lt_zero, val_max_gt_zero]), axis=0
                )

                # TODO: 1 difference in extreme conditions
                # assert (
                #    np.abs(
                #        len(np.where(val_abs_min == 0)[0])
                #        - (safe_ge_zero + safe_le_zero).cpu()
                #    )
                #    <= 1
                # )

                n = round(len(np.where(val_abs_min != 0)[0]) * self.intensity)
                self.logger.debug(
                    f"BS: fixed {n} out of {len(val_max_gt_zero)} neurons."
                )
                # n = 2
                n += safe_ge_zero + safe_le_zero - 1
                # print(n)
                pivot_value = val_abs_min[np.argsort(val_abs_min)[n]]
                # print(np.argsort(val_abs_min))
                # print(np.argsort(val_abs_min)[-n])
                # print(pivot_value)
                val_abs_min[val_abs_min > pivot_value] = 0
                # print(val_abs_min)

                a = np.where(val_min_lt_zero == val_abs_min)
                b = np.where(val_max_gt_zero == val_abs_min)
                # print(a)
                # print(b)

                x = np.zeros(val_min_lt_zero.shape)
                x[a[0]] = val_abs_min[a[0]]  # +SHIFT_EPSILON
                x *= -1
                x[b[0]] = val_abs_min[b[0]]  # +SHIFT_EPSILON
                x *= -1

                pretrained = self.model.state_dict()

                # self.logger.debug(f"BS: fixed {n} out of {len(x)} neurons.")
                idx = 0
                for i, (name, layer) in enumerate(self.model.filtered_named_modules):
                    old_bias = pretrained[f"{name}.bias"]

                    self.logger.debug(f"Old bias: {old_bias}")
                    x_ = x[idx : idx + len(old_bias)]

                    new_bias = old_bias.clone().cpu().detach().numpy() + x_
                    new_bias = torch.from_numpy(new_bias).to(
                        self.device, dtype=torch.float32
                    )
                    pretrained[f"{name}.bias"] = new_bias
                    self.model.load_state_dict(pretrained)
                    idx += len(old_bias)
                    self.logger.debug(f"New bias: {new_bias}")
                print(f"Fixed {n} neurons.")
            else:
                for i, (name, layer) in enumerate(self.model.filtered_named_modules):
                    # print(name, layer)

                    if any(isinstance(layer, x) for x in [nn.Linear, nn.Conv2d]):
                        self.stable_estimators.propagate(data=data)
                        val = self.model._batch_values[name]
                        le_0_, ge_0_ = self.stable_estimators.get_stable_ReLUs()
                        lb_, ub_ = self.stable_estimators.get_bounds()

                        if len(lb_[i].shape) == 2:
                            val_min = torch.min(lb_[i].T, axis=-1).values
                            val_max = torch.max(ub_[i].T, axis=-1).values
                        elif len(lb_[i].shape) == 4:
                            val_min = torch.amin(lb_[i], dim=(0, 2, 3))
                            val_max = torch.amax(ub_[i], dim=(0, 2, 3))
                        else:
                            raise ValueError(len(lb_[i].shape))

                        (
                            safe_le_zero,
                            safe_ge_zero,
                        ) = self.stable_estimators._calculate_stable_ReLUs(
                            val_min, val_max
                        )

                        val_min_lt_zero = np.copy(val_min.detach().cpu().numpy())
                        val_max_gt_zero = np.copy(val_max.detach().cpu().numpy())

                        if self.mode == "standard":
                            # pick lb < 0
                            val_min_lt_zero[val_min_lt_zero >= 0] = 0
                            # print(val_min_lt_zero)
                            val_min_lt_zero *= -1
                            # pick ub > 0
                            val_max_gt_zero[val_max_gt_zero <= 0] = 0
                            # print(val_max_gt_zero)

                            val_abs_min = np.min(
                                np.array([val_min_lt_zero, val_max_gt_zero]), axis=0
                            )

                            # TODO: 1 difference in extreme conditions
                            # assert (
                            #    np.abs(
                            #        len(np.where(val_abs_min == 0)[0])
                            #        - (safe_ge_zero + safe_le_zero).cpu()
                            #    )
                            #    <= 1
                            # )

                            n = round(
                                len(np.where(val_abs_min != 0)[0]) * self.intensity
                            )
                            self.logger.debug(f"BS: fixed {n} neurons.")
                            # n = 2
                            n += safe_ge_zero + safe_le_zero - 1
                            # print(n)
                            pivot_value = val_abs_min[np.argsort(val_abs_min)[n]]
                            # print(np.argsort(val_abs_min))
                            # print(np.argsort(val_abs_min)[-n])
                            # print(pivot_value)
                            val_abs_min[val_abs_min > pivot_value] = 0
                            # print(val_abs_min)

                            a = np.where(val_min_lt_zero == val_abs_min)
                            b = np.where(val_max_gt_zero == val_abs_min)
                            # print(a)
                            # print(b)

                            x = np.zeros(val_min.shape)
                            x[a[0]] = val_abs_min[a[0]]  # +SHIFT_EPSILON
                            x *= -1
                            x[b[0]] = val_abs_min[b[0]]  # +SHIFT_EPSILON
                            x *= -1

                            pretrained = self.model.state_dict()
                            fc1_bias = pretrained[f"{name}.bias"]
                            epsilon = 0.1
                            # print(fc1_bias.detach().numpy())
                            new_bias = fc1_bias.clone().cpu().detach().numpy() + x
                            # print(new_bias)

                            new_bias = torch.from_numpy(new_bias).to(
                                self.device, dtype=torch.float32
                            )
                            # model.fc1.bias = model.fc1.bias - new_bias
                            # model.fc1.bias = nn.Parameter(torch.randn(128))

                        elif self.mode == "distribution":
                            temp = torch.where(val.T <= 0)
                            nb_lt_zero = [
                                len(torch.where(temp[0] == x)[0])
                                for x in range(len(val.T))
                            ]
                            temp = torch.where(val.T > 0)
                            nb_gt_zero = [
                                len(torch.where(temp[0] == x)[0])
                                for x in range(len(val.T))
                            ]
                            a = np.mean(np.array(nb_lt_zero))
                            b = np.mean(np.array(nb_gt_zero))
                            lt += a
                            gt += b

                            shift = []
                            for i in range(len(val_min)):
                                if nb_lt_zero[i] >= nb_gt_zero[i]:
                                    if val_min[i] >= 0:
                                        print("ha")
                                        shift += [0]
                                    else:
                                        shift += [-val_min[i].numpy()]
                                else:
                                    if val_max[i] <= 0:
                                        print("ha")
                                        shift += [0]
                                    else:
                                        shift += [-val_max[i].numpy()]

                            temp = np.sort(np.abs(shift))[
                                : round(len(val.T) * self.intensity)
                            ]

                            new_shift = []
                            for x in shift:
                                if x in temp:
                                    new_shift += [x]
                                elif -x in temp:
                                    new_shift += [-x]
                                else:
                                    new_shift += [0]
                            print(new_shift)

                            pretrained = self.model.state_dict()
                            fc1_bias = pretrained[f"{name}.bias"]
                            new_bias = fc1_bias.clone().cpu().detach().numpy()

                            # print(new_bias)
                            new_bias += np.array(new_shift)
                            # print(new_bias)

                            new_bias = torch.from_numpy(new_bias).to(
                                self.device, dtype=torch.float32
                            )

                        pretrained[f"{name}.bias"] = new_bias
                        self.model.load_state_dict(pretrained)

                        # set model to eval mode
                        # calculate the new pre-activation values due to changes to this layer
                        self.model.eval()
                        with torch.no_grad():
                            self.model(data)
                    else:
                        print("ignored layer.")

                # reset model to train mode
                self.model.train()

        if BS_switch and self.model == "distribution":
            print(lt, gt)

        return BS_switch
