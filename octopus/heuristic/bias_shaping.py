import torch
import numpy as np

from . import Heuristic


class BiasShaping(Heuristic):
    def __init__(self, model, cfg):
        super().__init__(model.logger)
        self.model = model
        # self.cfg = cfg
        self.mode = cfg['mode']
        self.intensity = cfg['intensity']
        self.occurrence = cfg['occurrence'] if 'occurrence' in cfg else None
        self.every = cfg['every'] if 'every' in cfg else None
        self.decay = 1 if not 'decay' in cfg else cfg['decay']
        self.device = model.device

        assert not (self.occurrence and self.every)

    def run(self, **kwargs):
        epoch = kwargs['epoch']
        BS_switch = False

        # probability check
        if self.occurrence:
            BS_switch = np.random.rand() < self.occurrence * self.decay**epoch
        elif self.every:
            BS_switch = kwargs['batch'] % self.every == 0
        else:
            assert False
        gt = 0
        lt = 0
        if BS_switch:
            for layer in self.model.activation:
                val = self.model.activation[layer].view(self.model.activation[layer].size(0), -1)
                val_min = torch.min(val, axis=0).values
                val_max = torch.max(val, axis=0).values
                safe_ge_zero = torch.sum(val_min >= 0).int()
                safe_le_zero = torch.sum(val_max <= 0).int()

                val_min_lt_zero = np.copy(val_min.cpu().numpy())
                val_max_gt_zero = np.copy(val_max.cpu().numpy())

                if self.mode == 'standard':
                    # pick lb < 0
                    val_min_lt_zero[val_min_lt_zero > 0] = 0
                    # print(val_min_lt_zero)
                    val_min_lt_zero *= -1
                    # pick ub > 0
                    val_max_gt_zero[val_max_gt_zero < 0] = 0
                    # print(val_max_gt_zero)

                    val_abs_min = np.min(np.array([val_min_lt_zero, val_max_gt_zero]), axis=0)
                    # print(val_abs_min)
                    assert len(np.where(val_abs_min == 0)[0]) == safe_ge_zero + safe_le_zero

                    n = round(len(np.where(val_abs_min != 0)[0]) * self.intensity)
                    self.logger.debug(f'BS: fixed {n} neurons.')
                    #n = 2
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
                    fc1_bias = pretrained[f'{layer}.bias']
                    epsilon = 0.1
                    # print(fc1_bias.detach().numpy())
                    new_bias = fc1_bias.clone().cpu().detach().numpy() + x
                    # print(new_bias)

                    new_bias = torch.from_numpy(new_bias).to(self.device, dtype=torch.float32)
                    # model.fc1.bias = model.fc1.bias - new_bias
                    # model.fc1.bias = nn.Parameter(torch.randn(128))

                elif self.mode == 'distribution':
                    temp = torch.where(val.T <= 0)
                    nb_lt_zero = [len(torch.where(temp[0] == x)[0]) for x in range(len(val.T))]
                    temp = torch.where(val.T > 0)
                    nb_gt_zero = [len(torch.where(temp[0] == x)[0]) for x in range(len(val.T))]
                    a = np.mean(np.array(nb_lt_zero))
                    b = np.mean(np.array(nb_gt_zero))
                    lt += a
                    gt += b

                    shift = []
                    for i in range(len(val_min)):
                        if nb_lt_zero[i] >= nb_gt_zero[i]:
                            if val_min[i] >= 0:
                                print('haha')
                                shift += [0]
                            else:
                                shift += [-val_min[i].numpy()]
                        else:
                            if val_max[i] <= 0:
                                print('haha')
                                shift += [0]
                            else:
                                shift += [-val_max[i].numpy()]

                    temp = np.sort(np.abs(shift))[:round(len(val.T) * self.intensity)]

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
                    fc1_bias = pretrained[f'{layer}.bias']
                    new_bias = fc1_bias.clone().cpu().detach().numpy()

                    # print(new_bias)
                    new_bias += np.array(new_shift)
                    # print(new_bias)

                    new_bias = torch.from_numpy(new_bias).to(self.device, dtype=torch.float32)

                pretrained[f'{layer}.bias'] = new_bias
                self.model.load_state_dict(pretrained)

                # set model to eval mode
                # calculate the new pre-activation values due to changes to this layer
                self.model.eval()
                with torch.no_grad():
                    self.model(kwargs['data'])

            # reset model to train mode
            self.model.train()
        if BS_switch:
            print(lt, gt)
        return BS_switch
