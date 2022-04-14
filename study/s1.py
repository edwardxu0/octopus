train_nodes = None
train_nodes_ex = 'affogato12,affogato13,affogato14,cheetah01,ristretto01,ristretto02,ristretto03,ristretto04,lynx08,lynx09,ai04'

veri_nodes = ['cortado'+x for x in ['01', '02','03','04','05','06','07','08','09','10']]
veri_nodes_ex = None

artifacts = ['MNIST']
networks = {'NetS':[784, 128, 128, 128, 10],
            'NetM':[784, 1024, 1024, 1024, 10],
            'NetL':[784, 1024, 1024, 1024, 1024, 1024, 1024, 10]}

seeds = [*range(5)]

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
                  'base': None
                  }

props = [*range(5)]

epsilons = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

verifiers = ['neurify', 'eran_deepzono', 'nnenum']