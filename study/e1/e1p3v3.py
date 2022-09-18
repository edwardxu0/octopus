import numpy as np

sleep_time = 0

train_nodes = None
train_nodes_ex = "ai01,ai07,ai08,lynx08,lynx09,lynx10,sds01,sds02,lotus,titanx01,titanx02,titanx03,titanx04,titanx05,titanx06,lynx11,lynx12,affogato12,adriatic01,adriatic02,adriatic03,adriatic04,adriatic05,adriatic06"

veri_nodes = [
    "doppio" + x for x in ["01", "02", "03", "04", "05"]
]  # ,'06','07','08','09','10']]
veri_nodes_ex = None

artifacts = ["CIFAR10"]  # ["MNIST", "FashionMNIST", "CIFAR10"]

networks = {"NetL": [512] * 6}

seeds = [*range(3)]

heuristics = {
    "Baseline": None,
    # Bias Shaping
    "BS_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "pace": 100,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "BS_SAD": {
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
    },
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
            "pace": 400,
            "sparsity": 5e-2,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "PR_SAD": {
        "prune": {
            "mode": "stablenet",
            "pace": 400,
            "sparsity": 5e-2,
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
            "pace": 400,
            "sparsity": 5e-2,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
    "PR_SIP": {
        "prune": {
            "mode": "stablenet",
            "pace": 400,
            "sparsity": 5e-2,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
}

props = [*range(5)]

epsilons = np.linspace(2, 10, 5) / 100

verifiers = ["DNNV:eran_deeppoly", "DNNV:nnenum", "DNNVWB:neurify", "DNNV:marabou"]
