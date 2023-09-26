from . import *

artifacts = ["MNIST"]
networks = {"Net256x2": [256, 256]}

stabilizers = {
    "Baseline": None,
    # Bias Shaping
    "BS_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 200,
            "start": 1,
            "end": 50,
            "stable_estimators": {"SDD": {}},
        },
    },
    "BS_SAD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 200,
            "start": 1,
            "end": 50,
            "stable_estimators": {"SAD": {}},
        },
    },
    "BS_NIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 200,
            "start": 1,
            "end": 50,
            "stable_estimators": {"NIP": {}},
        },
    },
    "BS_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 200,
            "start": 1,
            "end": 50,
            "stable_estimators": {"SIP": {}},
        },
    },
    "BS_ALR": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 200,
            "start": 1,
            "end": 50,
            "stable_estimators": {"ALR": {"method": "CROWN"}},
        },
    },
    "BS_ALRo": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 200,
            "start": 1,
            "end": 50,
            "stable_estimators": {"ALR": {"method": "CROWN-optimized"}},
        },
    },
    # RS Loss
    "RS_SDD": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 50,
            "stable_estimators": {"SDD": {}},
        },
    },
    "RS_SAD": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 50,
            "stable_estimators": {"SAD": {}},
        },
    },
    "RS_NIP": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 50,
            "stable_estimators": {"NIP": {}},
        },
    },
    "RS_SIP": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 50,
            "stable_estimators": {"SIP": {}},
        },
    },
    "RS_ALR": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 50,
            "stable_estimators": {"ALR": {"method": "CROWN"}},
        },
    },
    "RS_ALRo": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 50,
            "stable_estimators": {"ALR": {"method": "CROWN-optimized"}},
        },
    },
    # PR
    "SP_SDD": {
        "stable_prune": {
            "mode": "standard",
            "pace": 200,
            "sparsity": 5e-2,
            "start": 1,
            "end": 50,
            "stable_estimators": {"SDD": {}},
        },
    },
    "SP_SAD": {
        "stable_prune": {
            "mode": "standard",
            "pace": 200,
            "sparsity": 5e-2,
            "start": 1,
            "end": 50,
            "stable_estimators": {"SAD": {}},
        },
    },
    "SP_NIP": {
        "stable_prune": {
            "mode": "standard",
            "pace": 200,
            "sparsity": 5e-2,
            "start": 1,
            "end": 50,
            "stable_estimators": {"NIP": {}},
        },
    },
    "SP_SIP": {
        "stable_prune": {
            "mode": "standard",
            "pace": 200,
            "sparsity": 5e-2,
            "start": 1,
            "end": 50,
            "stable_estimators": {"SIP": {}},
        },
    },
    "SP_ALR": {
        "stable_prune": {
            "mode": "standard",
            "pace": 200,
            "sparsity": 5e-2,
            "start": 1,
            "end": 50,
            "stable_estimators": {"ALR": {"method": "CROWN"}},
        },
    },
    "SP_ALRo": {
        "stable_prune": {
            "mode": "standard",
            "pace": 200,
            "sparsity": 5e-2,
            "start": 1,
            "end": 50,
            "stable_estimators": {"ALR": {"method": "CROWN-optimized"}},
        },
    },
}
