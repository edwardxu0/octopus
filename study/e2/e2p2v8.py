from study import *

artifacts = ["FashionMNIST"]  # ["MNIST", "FashionMNIST", "CIFAR10"]
networks = {"NetM": [1024] * 3}

heuristics = {
    # BS + RS
    "BR_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
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
    "BR_SAD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
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
    },
    "BR_NIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
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
    },
    "BR_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
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
    "BP_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
        "prune": {
            "mode": "stablenet",
            "pace": 800,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "BP_SAD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
        "prune": {
            "mode": "stablenet",
            "pace": 800,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
    },
    "BP_NIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
        "prune": {
            "mode": "stablenet",
            "pace": 800,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    "BP_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
        "prune": {
            "mode": "stablenet",
            "pace": 800,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
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
            "pace": 800,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "RP_SAD": {
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
            "mode": "stablenet",
            "pace": 800,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
    },
    "RP_NIP": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
        "prune": {
            "mode": "stablenet",
            "pace": 800,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
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
            "pace": 800,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    # BS + RS + PR
    "BRP_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
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
            "pace": 800,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "BRP_SAD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
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
            "mode": "stablenet",
            "pace": 800,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {
                "SAD": {"mode": "standard", "epsilon": 0.1, "samples": 100}
            },
        },
    },
    "BRP_NIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
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
            "mode": "stablenet",
            "pace": 800,
            "sparsity": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    "BRP_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-3,
            "pace": 800,
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
            "pace": 800,
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
