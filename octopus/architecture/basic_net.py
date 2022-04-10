import torch
import numpy as np
np.set_printoptions(suppress=True, precision=4)

from torch import nn


class BasicNet(nn.Module):
    def __init__(self, logger):
        super(BasicNet, self).__init__()
        self.logger = logger
    
    def __setup__(self):
        self._compute_filtered_named_modules()
        self.activation = self.register_activation_hocks(self)


    def _compute_filtered_named_modules(self):
        filtered_named_modules = []
        for name, module in self.named_modules():
            if 'dropout' not in name:
                filtered_named_modules += [(name,module)]
        filtered_named_modules = filtered_named_modules[1:-1]
        self.filtered_named_modules = filtered_named_modules


    def get_RS_loss(self, data, cfg):
        assert cfg['mode'] == 'standard'
        loss = torch.zeros((len(data),3))
        data = data.reshape((-1, 28*28))
        lb = torch.maximum(data - cfg['epsilon'], torch.tensor(0., requires_grad=True))
        ub = torch.minimum(data + cfg['epsilon'], torch.tensor(1., requires_grad=True))
            
        for i, (name, _) in enumerate(self.filtered_named_modules):
            w = self.__getattr__(name).weight
            b = self.__getattr__(name).bias
            lb, ub = self._interval_arithmetic(lb, ub, w, b)
            rs_loss = self._l_relu_stable(lb, ub)
            loss[i] +=  rs_loss
        loss = torch.sum(torch.mean(loss, axis=1))
        return loss

    # 
    def get_WD_loss(self, device):
        loss = torch.zeros(3).to(device=device)
        params = [14*14, 7*7, 1]
        c = 0
        for layer in self.state_dict():
            if 'weight' in layer and 'out' not in layer:
                w = self.state_dict()[layer]
                x = torch.abs(w)
                x = torch.sum(w) * params[c]
                loss[c] = x
                c += 1
        loss = torch.sum(loss)
        return loss


    """RS Loss Function"""
    def _l_relu_stable(self, lb, ub, norm_constant=1.0):
        loss = -torch.sum(torch.tanh(torch.tensor(1.0, requires_grad=True)+ norm_constant * lb * ub))
        return loss
    

    @staticmethod
    def _interval_arithmetic(lb, ub, W, b):
        W_max = torch.maximum(W, torch.tensor(0.0, requires_grad=True)).T
        W_min = torch.minimum(W, torch.tensor(0.0, requires_grad=True)).T
        new_lb = torch.matmul(lb, W_max) + torch.matmul(ub, W_min) + b
        new_ub = torch.matmul(ub, W_max) + torch.matmul(lb, W_min) + b
        return new_lb, new_ub


    @staticmethod
    def get_activation(name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    

    @staticmethod
    def register_activation_hocks(model):
        activation = {}
        for name, module in model.filtered_named_modules:
            module.register_forward_hook(model.get_activation(name,activation))
        return activation


    # restructure architecture to remove empty neurons
    def restructure(self):
        pass

    # pruning code ... 
    def prune(self, cfg):
        pass

    # Bias shaping
    def stable_ReLU(self):
        safe_le_zero_all = []
        safe_ge_zero_all = []
        for layer in self.activation.keys():
            val = self.activation[layer].view(self.activation[layer].size(0),-1).cpu().numpy()
            val_min = np.min(val, axis=0)
            val_max = np.max(val, axis=0)
            safe_ge_zero = (np.asarray(val_min) >= 0).sum()
            safe_le_zero = (np.asarray(val_max) <= 0).sum()

            safe_le_zero_all += [safe_le_zero]
            safe_ge_zero_all += [safe_ge_zero]
        safe_all = sum(safe_le_zero_all)+sum(safe_ge_zero_all)
        return safe_all


    def bias_shaping(self, cfg, data, device):
        BS_switch = False

        # probability check
        #if np.random.rand() < cfg['occurrence']:
        if True:    
            # bias shaping modes
            if cfg['mode'] == 'standard':
                BS_switch = True
            else:
                assert False

            if BS_switch:
                for layer in self.activation.keys():
                    val = self.activation[layer].view(self.activation[layer].size(0),-1)
                    val_min = torch.min(val, axis=0).values
                    val_max = torch.max(val, axis=0).values
                    safe_ge_zero = torch.sum(val_min >= 0).int()
                    safe_le_zero = torch.sum(val_max <= 0).int()

                    # assert safe_ge_zero == 0
                    val_min_lt_zero = np.copy(val_min.cpu())
                    val_max_gt_zero = np.copy(val_max.cpu())
                    
                    # pick lb < 0
                    val_min_lt_zero[val_min_lt_zero > 0] = 0
                    # print(val_min_lt_zero)
                    val_min_lt_zero*= -1
                    # pick ub > 0
                    val_max_gt_zero[val_max_gt_zero < 0] = 0
                    # print(val_max_gt_zero)
                    
                    val_abs_min = np.min(np.array([val_min_lt_zero, val_max_gt_zero]), axis=0)
                    #print(val_abs_min)
                    assert len(np.where(val_abs_min==0)[0]) == safe_ge_zero + safe_le_zero

                    n = round(len(np.where(val_abs_min!=0)[0]) * cfg['intensity'])
                    self.logger.info(f'BSed {n} neurons.')
                    #n = 2 
                    n += safe_ge_zero + safe_le_zero -1
                    #print(n)
                    pivot_value = val_abs_min[np.argsort(val_abs_min)[n]]
                    #print(np.argsort(val_abs_min))
                    #print(np.argsort(val_abs_min)[-n])
                    #print(pivot_value)
                    val_abs_min[val_abs_min > pivot_value] = 0
                    #print(val_abs_min)
                    
                    a = np.where(val_min_lt_zero == val_abs_min)
                    b = np.where(val_max_gt_zero == val_abs_min)
                    #print(a)
                    #print(b)
                    
                    x = np.zeros(val_min.shape)
                    x[a[0]] = val_abs_min[a[0]]#+SHIFT_EPSILON
                    x*=-1
                    x[b[0]] = val_abs_min[b[0]]#+SHIFT_EPSILON
                    x*=-1

                    pretrained = self.state_dict()
                    fc1_bias = pretrained[f'{layer}.bias']
                    epsilon = 0.1
                    #print(fc1_bias.detach().numpy())
                    new_bias = fc1_bias.clone().cpu().detach().numpy() + x
                    #print(new_bias)
                    
                    new_bias = torch.from_numpy(new_bias).to(device, dtype=torch.float32)
                    # model.fc1.bias = model.fc1.bias - new_bias
                    # model.fc1.bias = nn.Parameter(torch.randn(128))
                    pretrained[f'{layer}.bias'] = new_bias
                    self.load_state_dict(pretrained)

                    # set model to eval mode
                    # calculate the new pre-activation values due to changes to this layer
                    self.eval()
                    self(data)
                
                # reset model to train mode
                self.train()

        return BS_switch
