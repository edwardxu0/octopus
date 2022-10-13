from study import *

artifacts = ["CIFAR10"]  # ["MNIST", "FashionMNIST", "CIFAR10"]
networks = {"NetL": [512] * 6}

heuristics = {
    # BS + RS
    "BR_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 400,
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
    },
    # BS + PR
    "BP_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 400,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
        "prune": {
            "mode": "stablenet",
            "pace": 400,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    # RS + PR
    "RP_SIP": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
        "prune": {
            "mode": "stablenet",
            "pace": 400,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    # BS + RS + PR
    "BRP_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 400,
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
            "mode": "stablenet",
            "pace": 400,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    # Stabilizers
    # "SDD": {}
    # "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
    # "NIP": {"mode": "standard", "epsilon": 0.1}
    # "SIP": {"mode": "standard", "epsilon": 0.1}
}
