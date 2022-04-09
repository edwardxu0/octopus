#!/usr/bin/env python

import sys
from pathlib import Path

META_PATH = './meta/'
MODEL_DIR = './models/'
PROP_DIR = './properties/'
TRAIN_LOG_DIR = './raw_train/'
VERI_LOG_DIR = './raw_veri/'

from train_all import *
from analyze_train import *
from analyze_veri import *
from verify_all import *
from analyze_tv import *


def main():
    strat = 'baseline'
    # strat = 'rurh'
    strat = 'rs'
    # problems = ['MNIST', 'FashionMNIST']
    problems = ['MNIST']
    nets = ['NetS', 'NetM', 'NetL']
    max_safe_relus = {'NetS':128*3, 'NetM': 1024*3, 'NetL': 1024*6}

    itst = [None]
    # itst = [5]
    # itst = [None]

    # ocrc = [*range(0,11)]
    ocrc = [None]
    # ocrc = [0.005]
    # ocrc = [None]
    # upbd = [*range(50,101,10)]

    upbd = [None]
    # upbd = [100]
    # upbd = [None]
    
    seeds = [0] # [*range(10,15)]
    props = [*range(5)]

    epses = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

    verifiers = ['neurify','eran','nnenum']

    task = sys.argv[1]

    go = len(sys.argv) > 2 and sys.argv[2] == 'go'
    show = len(sys.argv) > 2 and sys.argv[2] == 'show'

    if task == 'T':
        Path(TRAIN_LOG_DIR).mkdir(parents=True, exist_ok=True)
        train_all(strat, problems, nets, itst, ocrc, upbd, seeds, slurm=True, go=go)

    elif task == 'TA':
        analyze_train(strat, problems, nets, itst, ocrc, upbd, seeds, max_safe_relus, show=show, debug=True)

    elif task == 'V':
        Path(VERI_LOG_DIR).mkdir(parents=True, exist_ok=True)
        verify_all(strat, problems, nets, itst, ocrc, upbd, props, epses, seeds, verifiers, slurm=True, wb=False, go=go)

    elif task == 'VA':
        analyze_veri(strat, problems, nets, itst, ocrc, upbd, props, epses, seeds, verifiers, show=show, debug=True)
    
    elif task == 'A':
        analyze_tv(strat, problems, nets, epses, verifiers, max_safe_relus)

    else:
        assert False


if __name__ == '__main__':
    main()
