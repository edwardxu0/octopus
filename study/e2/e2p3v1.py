from study import *

artifacts = ["CIFAR10"]  # ["MNIST", "FashionMNIST", "CIFAR10"]
networks = {"NetL": [512] * 6}

heuristics = {
    # BS + RS
    "BR_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 1e-2,
            "pace": 200,
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
    # BS + PR
    "BP_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 1e-2,
            "pace": 200,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
        "prune": {
            "mode": "stablenet",
            "pace": 200,
            "sparsity": 1e-2,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    # RS + PR
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
            "pace": 200,
            "sparsity": 1e-2,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    # BS + RS + PR
    "BRP_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 1e-2,
            "pace": 200,
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
            "pace": 200,
            "sparsity": 1e-2,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    # Stabilizers
    # "SDD": {},
    # "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100},
    # "NIP": {"mode": "standard", "epsilon": 0.1},
    # "SIP": {"mode": "standard", "epsilon": 0.1},
}
