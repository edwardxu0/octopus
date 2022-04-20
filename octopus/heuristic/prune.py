import torch
import copy

from . import Heuristic


class Prune(Heuristic):
    def __init__(self, model, cfg):
        self.model = model
        self.sparsity = cfg['sparsity']
        self.re_arch = cfg['re_arch']

    # pruning code ...

    def run(self, data):
        activation = data[0]
        mask = data[1]
        self.model.logger.info('prune starts here ...')
        # prune weights
        mask=self.update_mask(mask, activation, 0.2)
        self.model=self.apply_mask( mask)


        # restructure network
        if self.re_arch:
            self.restructure(mask)

    # restructure architecture to remove empty neurons

    def restructure(self,mask):
        # save weight bias here
        #w = self.model.state_dict().clone()

        # for l in self.model.filtered_named_modules:
        #     print(l)

        
        # compute new weights, bias
        weight, bias = self.make_param_matrix(mask)
        # clear model
        self.model.clear()
        # layers = [int(sum(mask[i])) for i in mask]
        # print("layers: ", layers)

        # construct new model
        # e.g.
        # layers = [784, 128, 10]
        # weights = [torch.FloatTensor(784, 128).uniform_().to(self.model.device),
        #            torch.FloatTensor(128, 128).uniform_().to(self.model.device),
        #            torch.FloatTensor(128, 128).uniform_().to(self.model.device),
        #            torch.FloatTensor(128, 10).uniform_().to(self.model.device)]
        # bias = [torch.FloatTensor(128).uniform_().to(self.model.device),
        #         torch.FloatTensor(128).uniform_().to(self.model.device),
        #         torch.FloatTensor(128).uniform_().to(self.model.device),
        #         torch.FloatTensor(10).uniform_().to(self.model.device)]

        self.model.set_layers(layers, weight, bias)

    def make_param_matrix(self, mask):
        '''
        Function for making weight matrix for reconstructed model
        input:model: from which reconstructed model will be made
              mask: matrix containing 0 for pruned and 1 for unpruned neurons of the model
        output: weight matrix: containing weights of all unpruned neurons
                bias matrix: containing bias of all unpruned neurons
        '''
        weight=[]
        bias = []

        for i in mask:
            lst_w = []
            lst_b = []
            

            for j in range(len(mask[i])):
                for name, param in self.model.named_parameters():
                    if name == f"{i}.weight" and mask[i][j]!=0:
                        lst_w.append(param[j].reshape(1,-1))
                        
                    elif name == f"{i}.bias" and mask[i][j]!=0:
                        lst_b.append(param[j])


            w=torch.cat(lst_w,dim=0)
            b=torch.tensor(lst_b)
            weight.append(w)
            bias.append(b)

        for name, param in self.model.named_parameters():
            temp=1
            for i in mask:
                if i in name:
                    temp=0
                    break

            if temp==1 and "weight" in name:
                weight.append(param)
            elif temp==1 and "bias" in name:
                bias.append(param)


        final_weight=[]
        final_weight.append(weight[0])

        l=1
        keys=[k for k in mask.keys()]
        while l<=len(mask):
            k=keys[l-1]
            lst_w = []
            for i in range(len(weight[l])):
                temp=[]
                t=weight[l][i]
                for j in range(len(mask[k])):
                    if mask[k][j]!=0:
                        temp.append(t[j])
                temp = torch.tensor(temp)
                lst_w.append(temp.reshape(1,-1))

            w = torch.cat(lst_w,dim=0)
            final_weight.append(w)

            l=l+1


        return final_weight, bias

    def apply_mask(self, mask):

        with torch.no_grad():            
            for name, module in self.model.filtered_named_modules:
                strg=f"self.model.{name}.weight"
                param = eval(strg)


                if 'conv' in strg:
                    param.copy_(mask[name].unsqueeze(1).unsqueeze(2).unsqueeze(3)*param)
                else:
                    param.copy_(mask[name].unsqueeze(1)*param)

                strg=f"self.model.{name}.bias"
                param = eval(strg)
                param.copy_(mask[name]*param)

        return self.model 

    def update_mask(self,mask, activation, pr_ratio):
        mean_activation={}
        for i in activation:
            if 'conv' in i:
                t=torch.mean(activation[i].abs(),0)
                #mean_activation[i]=torch.mean(torch.flatten(t))
                mean_activation[i]=torch.mean(torch.flatten(t,1),1)
            else:
                #mean_activation[i]=torch.mean(activation[i].abs())
                mean_activation[i]=torch.mean(activation[i].abs(),0)
        tf=mean_activation[next(iter(mean_activation))]
        i=0
        for key in mean_activation:
            if i!=0:
                tf=torch.hstack((tf,mean_activation[key]))
            i=i+1 
        tf,_=torch.sort(tf)
        tf = tf[tf.nonzero().squeeze()]
        nummask=0
        for v in mask.values():
            nummask+=torch.sum(v)
        threshold=tf[int(pr_ratio*nummask)]
        umask = copy.deepcopy(mask)
        for key in mean_activation:
            for i in range(len(mean_activation[key])):
                if mean_activation[key][i].abs()<threshold:
                    umask[key][i]=0
        return umask
