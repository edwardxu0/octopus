import numpy as np

sleep_time = 0
train_nodes = None
# train_nodes_ex = 'affogato11,affogato12,affogato13,affogato14,affogato15,cheetah01,lynx08,lynx09,lynx10,lynx11,lynx12,ai01,ai02,ai04,lotus,adriatic06'

# train_nodes_ex = "sds01,sds02,adriatic01,adriatic02,adriatic03,adriatic04,adriatic05,adriatic06,affogato12,affogato13,affogato14,affogato15,lotus"
# nb_train_nodes = 24
train_nodes_ex = "ai01,ai07,ai08,lynx08,lynx09,lynx10,sds01,sds02,lotus,titanx01,titanx02,titanx03,lynx11,lynx12,affogato12,adriatic01,adriatic02,adriatic03,adriatic04,adriatic05,adriatic06"
# train_nodes = ["cheetah01"]

veri_nodes = [
    "doppio" + x for x in ["01", "02", "03", "04", "05"]
]  # ,'06','07','08','09','10']]
veri_nodes_ex = None

artifacts = ["CIFAR10"]  # ["MNIST", "FashionMNIST", "CIFAR10"]

networks = {"NetL": [512] * 6}

seeds = [*range(1)]

heuristics = {
    # BS/RS
    "BRS_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 2e-2,
            "pace": 400,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-5,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
        "prune": {
            "mode": "structure",
            "re_arch": "last",
            "sparsity": 0.05,
            "start": 21,
            "end": 25,
            "stable_estimator": {"SDD": {}},
        },
    },
    "BRS_SAD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 2e-2,
            "pace": 400,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-5,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
        "prune": {
            "mode": "structure",
            "re_arch": "last",
            "sparsity": 0.05,
            "start": 21,
            "end": 25,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
    },
    "BRS_NIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 2e-2,
            "pace": 400,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-5,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
        "prune": {
            "mode": "structure",
            "re_arch": "last",
            "sparsity": 0.05,
            "start": 21,
            "end": 25,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    "BRS_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 2e-2,
            "pace": 400,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-5,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
        "prune": {
            "mode": "structure",
            "re_arch": "last",
            "sparsity": 0.05,
            "start": 21,
            "end": 25,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
}


# props = [*range(5)]
props = [*range(5)]

# epsilons = np.linspace(1, 10, 10) / 100
epsilons = np.linspace(2, 10, 5) / 100

verifiers = ["DNNV:eran_deeppoly", "DNNV:nnenum", "DNNVWB:neurify", "DNNV:marabou"]
