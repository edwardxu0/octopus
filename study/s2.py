sleep_time = 1
train_nodes = None
train_nodes_ex = 'affogato12,affogato13,affogato14,cheetah01,lynx08,lynx09,lynx11,lynx12,ai04'

veri_nodes = ['doppio'+x for x in ['01', '02', '03', '04', '05']]  # ,'06','07','08','09','10']]
veri_nodes_ex = None

artifacts = ['MNIST', 'FashionMNIST', 'CIFAR10']
networks = {'NetS': [128, 128, 128],
            'NetM': [1024, 1024, 1024],
            'NetL': [1024, 1024, 1024, 1024, 1024, 1024]}

seeds = [*range(3)]

heuristics = {'bias_shaping': {
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
    'base': None
}

props = [*range(5)]

epsilons = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

verifiers = ['neurify', 'eran_deepzono', 'nnenum']
