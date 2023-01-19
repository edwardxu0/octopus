from study_10 import *

artifacts = ["MNIST"]  # ["MNIST", "FashionMNIST", "CIFAR10"]
networks = {"FC4": [256] * 4}

heuristics = {
    "Baseline": None,
    # Bias Shaping
    "BR_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
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
    },
    "BP_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 100,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
        "prune": {
            "mode": "stablenet",
            "pace": 100,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "RP_SDD": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
        "prune": {
            "mode": "stablenet",
            "pace": 100,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "BRP_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
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
            "mode": "stablenet",
            "pace": 100,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
}
