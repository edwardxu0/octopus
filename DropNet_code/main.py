import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import numpy as np
import copy
import csv
import os
import time
import argparse

from DropNet import prune
#from Neuron_merging import *
#from Lookahead_pruning import *







if __name__=='__main__':
    # settings
    parser = argparse.ArgumentParser(description='DropNet Example')
    parser.add_argument('--pruning_method', action='store', default='DropNet', help='The pruning methods: DropNet|OpenLth|Neuron_merging|Lookahed')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
            help='input batch size for training and validating (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
            help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
            help='learning rate (default: 0.001)')
   
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--network', action='store', default='NetS',
            help='network structure: NetS|NetM|NetL'),
    parser.add_argument('--dataset', action='store', default='mnist',
            help='dataset: mnist')
    parser.add_argument('--name', default='NetS',
            help='output folder name')
    parser.add_argument('--pruning_ratio', type=float, default=0.2,
            help='pruning ratio : (default: 0.2)')
    parser.add_argument('--tolerance', type=float, default=0.05,
            help='how much accuracy drop will be tolerated : (default: 0.01)')
    parser.add_argument('--layers', type=int, nargs='+', default=[500, 500, 500, 500, 500, 500])



    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    
    

    #Dataset processing
    
    if args.pruning_method=='DropNet':
        prune(args)

    












