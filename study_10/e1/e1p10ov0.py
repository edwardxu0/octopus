from study_10 import *


artifacts = ["CIFAR10"]  # ["MNIST", "FashionMNIST", "CIFAR10"]
networks = {"OVAL21_o": [1]}

heuristics = {
    "Baseline": None,
    # Bias Shaping
    "BS_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 100,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "BS_SAD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
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
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "PR_SAD": {
        "prune": {
            "mode": "stablenet",
            "pace": 100,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
    },
}
