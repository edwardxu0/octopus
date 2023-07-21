from study_pre import *

artifacts = ["MNIST"]
networks = {"Net256x2": [256, 256]}

stabilizers = {
    "BS_ALR2": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
            "start": 1,
            "end": 50,
            "stable_estimators": {"ALR": {"method": "CROWN-optimized"}},
        },
    },
    "RS_ALR2": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 50,
            "stable_estimators": {"ALR": {"method": "CROWN-optimized"}},
        },
    },
    "SP_ALR2": {
        "stable_prune": {
            "mode": "standard",
            "pace": 100,
            "sparsity": 5e-2,
            "start": 1,
            "end": 50,
            "stable_estimators": {"ALR": {"method": "CROWN-optimized"}},
        },
    },
}
