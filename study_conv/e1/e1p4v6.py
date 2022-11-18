from study_conv import *

artifacts = ["CIFAR10"]  # ["MNIST", "FashionMNIST", "CIFAR10"]
networks = {"LeNet2": [1]}

heuristics = {
    "Baseline": None,
    # Bias Shaping
    "BS_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 1e-1,
            "pace": 100,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "BS_SAD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 1e-1,
            "pace": 100,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
    },
    # RS Loss
    "RS_SDD": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "RS_SAD": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
    },
    # PR
    "PR_SDD": {
        "prune": {
            "mode": "stablenet",
            "pace": 100,
            "sparsity": 1e-1,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "PR_SAD": {
        "prune": {
            "mode": "stablenet",
            "pace": 100,
            "sparsity": 1e-1,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
    },
}

new = {
    "BS_NIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    "BS_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    # RS Loss
    "RS_SDD": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "RS_SAD": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
    },
    "RS_NIP": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    "RS_SIP": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    # Pruning
    "PR_SDD": {
        "prune": {
            "mode": "stablenet",
            "pace": 100,
            "sparsity": 1e-2,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "PR_SAD": {
        "prune": {
            "mode": "stablenet",
            "pace": 100,
            "sparsity": 1e-2,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
    },
    "PR_NIP": {
        "prune": {
            "mode": "stablenet",
            "pace": 100,
            "sparsity": 1e-2,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    "PR_SIP": {
        "prune": {
            "mode": "stablenet",
            "pace": 100,
            "sparsity": 1e-2,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
}

"""
heuristics = {
    # RS Loss
    "RS_SDD": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "RS_SAD": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
    },
    "RS_NIP": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    "RS_SIP": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
    }
}
"""
