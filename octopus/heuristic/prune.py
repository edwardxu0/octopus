import torch

from . import Heuristic

class Prune(Heuristic):
    def __init__(self, model, cfg):
        self.model = model
        self.sparsity = cfg['sparsity']
        self.re_arch = cfg['restructure']


    # pruning code ... 
    def run(self, data):
        self.model.logger.info('prune starts here ...')
        # prune weights

        ...

        # restructure network
        if self.re_arch:
            self.restructure()


    # restructure architecture to remove empty neurons
    def restructure(self):
        # save weight bias here
        w = self.model.state_dict().clone()

        for l in self.model.filtered_named_modules:
            print(l)
        # clear model
        self.model.clear()
        # compute new weights, bias
        ...
        

        # construct new model
        # e.g.
        layers = [784,128,10]
        weights = [torch.FloatTensor(784,128).uniform_().to(self.model.device),
                    torch.FloatTensor(128,128).uniform_().to(self.model.device),
                    torch.FloatTensor(128,128).uniform_().to(self.model.device),
                    torch.FloatTensor(128,10).uniform_().to(self.model.device)]
        bias = [torch.FloatTensor(128).uniform_().to(self.model.device),
                torch.FloatTensor(128).uniform_().to(self.model.device),
                torch.FloatTensor(128).uniform_().to(self.model.device),
                torch.FloatTensor(10).uniform_().to(self.model.device)]

        self.model.set_layers(layers, weights, bias)