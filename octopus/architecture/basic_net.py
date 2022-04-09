import torch
import numpy as np

from torch import nn


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        
    
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


    def get_RS_loss(self, data, cfg_rs_loss):
        loss = torch.zeros((len(data),3))
        data = data.reshape((-1, 28*28))
        lb = torch.maximum(data - cfg_rs_loss['epsilon'], torch.tensor(0., requires_grad=True))
        ub = torch.minimum(data + cfg_rs_loss['epsilon'], torch.tensor(1., requires_grad=True))
            
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
            val = self.activation[layer].view(self.activation[layer].size(0),-1).numpy()
            val_min = np.min(val, axis=0)
            val_max = np.max(val, axis=0)
            safe_ge_zero = (np.asarray(val_min) >= 0).sum()
            safe_le_zero = (np.asarray(val_max) <= 0).sum()

            safe_le_zero_all += [safe_le_zero]
            safe_ge_zero_all += [safe_ge_zero]
        safe_all = sum(safe_le_zero_all)+sum(safe_ge_zero_all)
        return safe_all


    def bias_shaping(self, cfg_bs,data, epoch, batch_idx, device):
        f_rurh = False

        safe_le_zero_all = []
        safe_ge_zero_all = []
        for layer in self.activation.keys():
            val = self.activation[layer].view(self.activation[layer].size(0),-1).numpy()
            val_min = np.min(val, axis=0)
            val_max = np.max(val, axis=0)
            safe_ge_zero = (np.asarray(val_min) >= 0).sum()
            safe_le_zero = (np.asarray(val_max) <= 0).sum()

            safe_le_zero_all += [safe_le_zero]
            safe_ge_zero_all += [safe_ge_zero]

        safe_all = sum(safe_le_zero_all)+sum(safe_ge_zero_all)
        max_safe_relu = sum([self.activation[layer].view(self.activation[layer].size(0),-1).shape[-1] for layer in self.activation])
        
        # When is RURH activated?
        # 1) don't happen on first mini batch
        # 1) rurh is on
        # 2) last epochs is excepted
        # 3) occurance > 0
        # 4) ocrc% of time
        # 5) doesnt exceed upper bound
        if batch_idx > 0 and np.random.rand() < cfg_bs['occurrence']:
            
            if cfg_bs['type'] == 'standard':
                f_rurh = True
            else:
                assert False

            if f_rurh:
                for layer in self.activation.keys():
                    self.eval()
                    self(data)

                    val = self.activation[layer].view(self.activation[layer].size(0),-1).numpy()
                    val_min = np.min(val, axis=0)
                    val_max = np.max(val, axis=0)
                    safe_ge_zero = (np.asarray(val_min) >= 0).sum()
                    safe_le_zero = (np.asarray(val_max) <= 0).sum()

                    # assert safe_ge_zero == 0
                    val_min_lt_zero = np.copy(val_min)
                    val_max_gt_zero = np.copy(val_max)
                    
                    # pick lb < 0
                    val_min_lt_zero[val_min_lt_zero > 0] = 0
                    # print(val_min_lt_zero)
                    val_min_lt_zero*= -1
                    # pick ub > 0
                    val_max_gt_zero[val_max_gt_zero < 0] = 0
                    # print(val_max_gt_zero)

                    val_abs_min = np.min(np.array([val_min_lt_zero, val_max_gt_zero]), axis=0)
                    len(np.where(val_abs_min==0)[0]) == safe_ge_zero + safe_le_zero

                    n = int(len(np.where(val_abs_min!=0)[0]) * cfg_bs['intensity'])
                    
                    pivot_value = val_abs_min[np.argsort(val_abs_min)[-n]]

                    val_abs_min[val_abs_min < pivot_value] = 0

                    a = np.where(val_min_lt_zero == val_abs_min)
                    b = np.where(val_max_gt_zero == val_abs_min)
                    x = np.zeros(val_min.shape)
                    x[a[0]] = val_abs_min[a[0]]#+SHIFT_EPSILON
                    x*=-1
                    x[b[0]] = val_abs_min[b[0]]#+SHIFT_EPSILON
                    x*=-1

                    pretrained = self.state_dict()
                    fc1_bias = pretrained[f'{layer}.bias']
                    epsilon = 0.1
                    new_bias = fc1_bias.detach().numpy() + x
                    new_bias = torch.from_numpy(new_bias).to(device, dtype=torch.float32)
                    # model.fc1.bias = model.fc1.bias - new_bias
                    # model.fc1.bias = nn.Parameter(torch.randn(128))
                    pretrained[f'{layer}.bias'] = new_bias
                    self.load_state_dict(pretrained)

        if f_rurh:
            safe_le_zero_all = []
            safe_ge_zero_all = []
            for layer in self.activation.keys():
                val = self.activation[layer].view(self.activation[layer].size(0),-1).numpy()
                val_min = np.min(val, axis=0)
                val_max = np.max(val, axis=0)
                safe_ge_zero = (np.asarray(val_min) >= 0).sum()
                safe_le_zero = (np.asarray(val_max) <= 0).sum()

                safe_le_zero_all += [safe_le_zero]
                safe_ge_zero_all += [safe_ge_zero]
        
        return f_rurh, sum(safe_le_zero_all), sum(safe_ge_zero_all)