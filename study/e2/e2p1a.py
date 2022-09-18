import numpy as np

sleep_time = 5
train_nodes = None
# train_nodes_ex = 'affogato11,affogato12,affogato13,affogato14,affogato15,cheetah01,lynx08,lynx09,lynx10,lynx11,lynx12,ai01,ai02,ai04,lotus,adriatic06'

# train_nodes_ex = "sds01,sds02,adriatic01,adriatic02,adriatic03,adriatic04,adriatic05,adriatic06,affogato12,affogato13,affogato14,affogato15,lotus"
# nb_train_nodes = 24
train_nodes_ex = "ai01,ai07,ai08,lynx08,lynx09,lynx10,sds01,sds02,lotus,titanx01,titanx02,titanx03,lynx11,lynx12,affogato12"
# train_nodes = ["cheetah01"]

veri_nodes = [
    "doppio" + x for x in ["01", "02", "03", "04", "05"]
]  # ,'06','07','08','09','10']]
veri_nodes_ex = None

artifacts = ["MNIST"]  # ["MNIST", "FashionMNIST", "CIFAR10"]

# networks = {"NetS": [128] * 3, "NetM": [1024] * 3, "NetL": [1024] * 6}

networks = {"NetS": [128] * 3}

seeds = [*range(3)]

heuristics = {
    # BS + PR
    "BSPR_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
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
    "BSPR_SAD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
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
    "BSPR_NIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
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
    "BSPR_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
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
    # RS + PR
    "RSPR_SDD": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
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
    "RSPR_SAD": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
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
    "RSPR_NIP": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
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
    "RSPR_SIP": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
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
    # BS/RS + PR
    "BRSPR_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
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
    "BRSPR_SAD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
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
    "BRSPR_NIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
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
    "BRSPR_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
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