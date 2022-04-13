#!/usr/bin/env python
import sys
import toml
from train_all import *
from verify_all import *


def main():
    problems = ['MNIST']
    nets = {'NetS':[784, 128, 128, 128, 10],
            'NetM':[784, 1024, 1024, 1024, 10],
            'NetL':[784, 1024, 1024, 1024, 1024, 1024, 1024, 10]}
    
    seeds = [0] # [*range(10,15)]

    heuristics = {'bias_shaping':{
                                'mode':'standard',
                                'intensity': 5e-2,
                                'occurrence': 5e-3,
                                'start': 1,
                                'end': 20
                                },
                  'rs_loss':{
                                'mode':'standard',
                                'weight': 1e-4,
                                'epsilon': 1e-1,
                                'start': 1,
                                'end': 20
                                },
                  }

    props = [*range(5)]

    epsilons = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

    verifiers = ['neurify','eran','nnenum']

    if len(sys.argv) <= 2:
        print('./scripts/run.py base_toml task [go]')
        return

    task = sys.argv[2]
    go = len(sys.argv) > 3 and sys.argv[3] == 'go'
    config_file = open(sys.argv[1], 'r').read()
    settings = toml.loads(config_file)

    if task == 'T':
        train_all(settings, problems, nets, heuristics, seeds, slurm=True, go=go)

    elif task == 'TA':
        analyze_train()

    elif task == 'V':
        verify_all(settings, problems, nets, heuristics, seeds, props, epsilons, verifiers, slurm=True, go=go)

    elif task == 'VA':
        analyze_veri()
    
    elif task == 'A':
        analyze()
    else:
        assert False


if __name__ == '__main__':
    main()
