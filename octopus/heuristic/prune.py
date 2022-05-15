import torch
import copy

from . import Heuristic


class Prune(Heuristic):
    def __init__(self, model, cfg):
        super().__init__(model.logger)
        self.model = model
        self.mode = cfg['mode']
        self.sparsity = cfg['sparsity']
        self.re_arch = None if 're_arch' not in cfg else cfg['re_arch']

    # pruning code ...
    def run(self, **kwargs):
        self.logger.info('Prune starts here ...')
        # prune weights
        if self.mode == 'structure':
            self.logger.info('Using iterative structure pruning ...')
            mask = self._init_mask()
            mask = self._update_mask(mask, self.sparsity)
            self._apply_mask(mask)
        else:
            raise NotImplementedError(f'Prune mode: {self.mode} is not supported.')

        # restructure network
        if self.re_arch:
            if self.re_arch == 'standard':
                on = True
            elif self.re_arch == 'last' and kwargs['epoch'] == kwargs['total_epoch']:
                on = True
            else:
                on = False

            if on:
                self.logger.info('Restructuring network ...')
                self._restructure(mask)

    # initialize the mask
    def _init_mask(self):
        mask = {}
        for name, module in self.model.filtered_named_modules:
            if 'conv' in name:
                mname = 'model.'+upname+'.out_channels'
                j = module.out_channels
                mask[name] = torch.ones(j).to(self.model.device)
            elif 'FC' in name:
                j = module.out_features
                mask[name] = torch.ones(j).to(self.model.device)
        return mask

    def _update_mask(self, mask, pr_ratio):
        mean_activation = {}
        for i in self.model.activation:
            if 'conv' in i:
                t = torch.mean(self.model.activation[i].abs(), 0)
                mean_activation[i] = torch.mean(torch.flatten(t, 1), 1)
            else:
                mean_activation[i] = torch.mean(self.model.activation[i].abs(), 0)
        tf = mean_activation[next(iter(mean_activation))]
        i = 0
        for key in mean_activation:
            if i != 0:
                tf = torch.hstack((tf, mean_activation[key]))
            i = i+1
        tf, _ = torch.sort(tf)
        tf = tf[tf.nonzero().squeeze()]
        nummask = 0
        for v in mask.values():
            nummask += torch.sum(v)
        threshold = tf[int(pr_ratio*nummask)]
        umask = copy.deepcopy(mask)
        for key in mean_activation:
            for i in range(len(mean_activation[key])):
                if mean_activation[key][i].abs() < threshold:
                    umask[key][i] = 0
        return umask

    def _apply_mask(self, mask):
        with torch.no_grad():
            for name, module in self.model.filtered_named_modules:
                strg = f"self.model.{name}.weight"
                param = eval(strg)

                if 'conv' in strg:
                    param.copy_(mask[name].unsqueeze(1).unsqueeze(2).unsqueeze(3)*param)
                else:
                    param.copy_(mask[name].unsqueeze(1)*param)

                strg = f"self.model.{name}.bias"
                param = eval(strg)
                param.copy_(mask[name]*param)

    # restructure architecture to remove empty neurons
    def _restructure(self, mask):
        # compute new weights/bias
        weight, bias = self._make_param_matrix(mask)
        self.logger.debug(f'Remaining # neurons: {sum([b.shape[0] for b in bias])}')
        layers = [x.shape[0] for x in bias][:-1]
        # clear model
        self.model.clear()
        # construct new model
        self.model.set_layers(layers, weight, bias)
        self.logger.info(f'Restructured model: \n{self.model}')
        self.logger.info(f"# ReLUs: {self.model.nb_ReLUs}")
        self.model.to(self.model.device)

    def _make_param_matrix(self, mask):
        '''
        Function for making weight matrix for reconstructed model
        input:model: from which reconstructed model will be made
              mask: matrix containing 0 for pruned and 1 for unpruned neurons of the model
        output: weight matrix: containing weights of all unpruned neurons
                bias matrix: containing bias of all unpruned neurons
        '''
        weight = []
        bias = []

        for i in mask:
            lst_w = []
            lst_b = []

            for j in range(len(mask[i])):
                for name, param in self.model.named_parameters():
                    if name == f"{i}.weight" and mask[i][j] != 0:
                        lst_w.append(param[j].reshape(1, -1))

                    elif name == f"{i}.bias" and mask[i][j] != 0:
                        lst_b.append(param[j])

            w = torch.cat(lst_w, dim=0)
            b = torch.tensor(lst_b)
            weight.append(w)
            bias.append(b)

        for name, param in self.model.named_parameters():
            temp = 1
            for i in mask:
                if i in name:
                    temp = 0
                    break

            if temp == 1 and "weight" in name:
                weight.append(param)
            elif temp == 1 and "bias" in name:
                bias.append(param)

        final_weight = []
        final_weight.append(weight[0])

        l = 1
        keys = [k for k in mask.keys()]
        while l <= len(mask):
            k = keys[l-1]
            lst_w = []
            for i in range(len(weight[l])):
                temp = []
                t = weight[l][i]
                for j in range(len(mask[k])):
                    if mask[k][j] != 0:
                        temp.append(t[j])
                temp = torch.tensor(temp)
                lst_w.append(temp.reshape(1, -1))

            w = torch.cat(lst_w, dim=0)
            final_weight.append(w)

            l = l+1

        return final_weight, bias
