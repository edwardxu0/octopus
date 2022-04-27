import numpy as np

sleep_time = 5
train_nodes = None
train_nodes_ex = 'affogato12,affogato13,affogato14,affogato15,cheetah01,lynx08,lynx09,lynx10,lynx11,lynx12,ai01,ai02,ai04,lotus'

veri_nodes = ['doppio'+x for x in ['01', '02', '03', '04', '05']]  # ,'06','07','08','09','10']]
veri_nodes_ex = None

artifacts = ['MNIST', 'FashionMNIST', 'CIFAR10']

networks = {
    'NetS': [128]*3,
    'NetM': [1024]*3,
    'NetL': [1024]*6}

seeds = [*range(5)]

heuristics = {
    'bias_shaping': {
        'mode': 'standard',
        'intensity': 5e-2,
        'occurrence': 5e-3,
        'start': 1,
        'end': 100
    },
    'rs_loss': {
        'mode': 'standard',
        'weight': 1e-4,
        'epsilon': 1e-1,
        'start': 1,
        'end': 100
    },
    'prune': {
        'mode': 'structure',
        're_arch': 'standard',
        'sparsity': 0.05,
        'start': [*range(10, 91, 20)],
        'end': [*range(10, 91, 20)]
    },
    'base': None
}

props = [*range(3)]

epsilons = np.linspace(2, 20, 10)/100

verifiers = ['neurify', 'eran_deepzono', 'nnenum']
