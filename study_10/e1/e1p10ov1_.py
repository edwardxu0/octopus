from study_10 import *


artifacts = ["CIFAR10"]
networks = {"OVAL21_o": [1]}

heuristics = {
    "BS_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 1e-2,
            "pace": 200,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
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
    "PR_SIP": {
        "prune": {
            "mode": "stablenet",
            "pace": 200,
            "sparsity": 1e-2,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
}
