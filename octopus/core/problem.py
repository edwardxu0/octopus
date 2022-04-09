from re import A
import torch
import os
import numpy as np

from torch import optim
from torch.optim.lr_scheduler import StepLR

from ..datasets.dataset import gen_data_loader
from ..plot.train_progress import ProgressPlot
from ..architecture.architecture import *


class Problem():
    def __init__(self, settings):
        self.settings = settings
        self.cfg_train = settings.cfg_train
        self.cfg_relu_est = settings.cfg_relu_est
        self.cfg_heuristic = settings.cfg_heuristic
        self.cfg_verify = settings.cfg_verify
        self.logger = settings.logger
        self.setup(settings)

    def setup(self, settings):
        self.seed = settings.seed
        torch.manual_seed(settings.seed)
        np.random.seed(settings.seed)
        torch.set_printoptions(threshold=100000)

        # train data
        self.train_stable_ReLUs = []
        self.train_BS_points = []
        self.train_loss = []

        # test data
        self.test_accuracy = []

        self.model_name = self.get_model_name()
        
    
    def get_model_name(self):
        name =  f"A={self.cfg_train['artifact']}_N={self.cfg_train['network']}"
        name += f"_RE={self.cfg_relu_est['method']}"

        for h in self.cfg_heuristic:
            x = self.cfg_heuristic[h]

            if h == 'bias_shaping':
                if x['type'] == 'standard':
                    t = 'S'
                else:
                    raise NotImplementedError
                name += f"_BS={t}:{x['intensity']}:{x['occurrence']}:{x['start']}:{x['end']}"

            elif h == 'pruning':    
                if x['type'] == 'structure':
                    t = 'S'
                else:
                    raise NotImplementedError
                if x['re_arch'] == 'standard':
                    r = 'S'
                else:
                    raise NotImplementedError
                name += f"_PR={t}:{r}:{x['sparsity']}:{x['start']}:{x['end']}"

            elif h == 'rs_loss':
                if x['type'] == 'standard':
                    t = 'S'
                else:
                    raise NotImplementedError
                name += f"_RS={t}:{x['weight']}:{x['start']}:{x['end']}"
            else:
                assert False
        name += f'_seed={self.seed}'
        return name
    

    def train_setup(self):
        cfg = self.cfg_train
        use_gpu = True if 'gpu' in cfg and cfg['gpu'] else False
        use_cuda = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = gen_data_loader(cfg['artifact'], cfg['adv_train'], cfg['batch_size'], cfg['test_batch_size'], self.device)

        self.model = globals()[cfg['network']]().to(self.device)


    def train(self):
        self.train_setup()

        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=args.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg_train['lr'])
        scheduler = StepLR(self.optimizer, step_size=1, gamma=self.cfg_train['gamma'])

        for epoch in range(1, self.cfg_train['epochs'] + 1):
            self.train_epoch(epoch)
            self.test_epoch(epoch)

            # pruning
            if 'prune' in self.cfg_heuristic\
                and self.cfg_heuristic['prune']['start'] <= epoch <= self.cfg_heuristic['prune']['end']:

                self.model.prune(self.cfg_heuristic['prune'])

            if self.cfg_train['save_model']:
                dummy_input = torch.randn(1, 1, 28, 28, device=self.device)
                torch.onnx.export(self.model, dummy_input, os.path.join(self.settings.model_dir, f"{self.model_name}.onnx"), verbose=False)
                torch.save(self.model.state_dict(), os.path.join(self.settings.model_dir, f"{self.model_name}.pt"))
            
            scheduler.step()
        
        self.plot_train()


    def train_epoch(self, epoch):
        nb_safe_relu = []
        hammer_points = []
        losses = []

        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output_pre_softmax = self.model(data)
            output = F.log_softmax(output_pre_softmax, dim=1)

            loss = F.nll_loss(output, target)
            
            if 'rs_loss' in self.cfg_heuristic\
                and self.cfg_heuristic['rs_loss']['start'] <= epoch <= self.cfg_heuristic['rs_loss']['end']:

                cfg_rs_loss = self.cfg_heuristic['rs_loss']
                rs_loss = self.model.get_RS_loss(data, cfg_rs_loss)
                loss += rs_loss * cfg_rs_loss['weight']
            
            
            loss.backward()
            self.optimizer.step()

            # activate bias shaping
            if 'bias_shaping' in self.cfg_heuristic\
                and self.cfg_heuristic['bias_shaping']['start'] <= epoch <= self.cfg_heuristic['bias_shaping']['end']:
                self.model.bias_shaping(self.cfg_heuristic['bias_shaping'], data, epoch, batch_idx, self.device)
            #f_rurh, safe_le_zero, safe_ge_zero = bias_shaping(args, data, activation, model, device, epoch, batch_idx, test_accuracy, last_pre_rurh_accuracy)
            #if f_rurh:
            #    hammer_points += [len(train_loader)*(epoch-1) +batch_idx]
            #nb_safe_relu += [safe_le_zero+safe_ge_zero]

            self.train_stable_ReLUs += [self.model.stable_ReLU()]
            self.train_loss += [loss.item()]

            if batch_idx % self.cfg_train['log_interval'] == 0:
                #print('Train Epoch: {} batch: {} {:5.2f}% Loss: {:10.6f}'.format(epoch, batch_idx,
                #    100. * batch_idx / len(train_loader), loss.item()),end='')
                #print(f'\t{f_rurh}', end='')
                # print(f'\tSafe ReLU: {safe_le_zero+safe_ge_zero:6.0f}: (-: {safe_le_zero:6.0f}, +: {safe_ge_zero:6.0f})')

                print(f'Train epoch: {epoch} batch: {batch_idx:5} {100.*batch_idx/len(self.train_loader):5.2f} Loss: {loss.item():10.6f} ')# RURH: {str(f_rurh)[0]} Safe ReLU: {safe_le_zero+safe_ge_zero:6.0f}: (-: {safe_le_zero:6.0f}, +: {safe_ge_zero:6.0f})')
        

    def test_epoch(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output = F.log_softmax(output, dim=1)
                
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        test_accuracy = correct / len(self.test_loader.dataset)
        self.test_accuracy += [test_accuracy]

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * test_accuracy))
        return correct/ len(self.test_loader.dataset)


    def plot_train(self):
        # draw training progress plot
        X1 = range(len(self.train_stable_ReLUs))
        Y1 = self.train_stable_ReLUs
        X2 = self.train_BS_points
        Y2 = np.array(Y1)[self.train_BS_points]
        X3 = (np.array(range(len(self.test_accuracy)))+1)*len(self.train_loader)
        Y3 = self.test_accuracy
        X4 = range(len(self.train_loss))
        Y4 = self.train_loss

        max_safe_relu = sum([self.model.activation[layer].view(self.model.activation[layer].size(0),-1).shape[-1] for layer in self.model.activation])
        
        p_plot = ProgressPlot()
        p_plot.draw_train(X1, Y1, X2, Y2, (0, max_safe_relu))
        p_plot.draw_accuracy(X3, Y3, X4, Y4, (0,1))

        title =  f'# ...'
        path = f'{self.settings.result_dir}/{self.model_name}.png'
        p_plot.save(title, path)


    def try_prop(model, device, test_loader, args, activation, epoch, test_accuracy, last_pre_rurh_accuracy):
        model.eval()
        eps = 0.2
        safe = []
        with torch.no_grad():
            for data, target in test_loader:
                data = [data[0]]
                for i, d in enumerate(data):
                    d_ = []
                    d_ = np.random.uniform(d.numpy()[0]-eps, d.numpy()[0]+eps, size = (1000,28,28))
                    d_ = torch.from_numpy(d_).to(device, dtype=torch.float32)
                    model(d_)
                    f_rurh, safe_le_zero, safe_ge_zero = bias_shaping(args, data, activation, model, device, epoch, i, test_accuracy, last_pre_rurh_accuracy)
                    safe += [safe_le_zero+ safe_ge_zero]
        # safe = np.mean(np.array(safe))
        print(f'Per property Safe ReLUs: {safe}')