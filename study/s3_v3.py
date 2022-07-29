import numpy as np

sleep_time = 4
train_nodes = None
# train_nodes_ex = 'affogato11,affogato12,affogato13,affogato14,affogato15,cheetah01,lynx08,lynx09,lynx10,lynx11,lynx12,ai01,ai02,ai04,lotus,adriatic06'

train_nodes_ex = (
    "sds01,sds02,adriatic01,adriatic02,adriatic03,adriatic04,adriatic05,adriatic06"
)
train_nodes_ex = None
train_nodes = ["cheetah01"]

veri_nodes = [
    "doppio" + x for x in ["01", "02", "03", "04", "05"]
]  # ,'06','07','08','09','10']]
veri_nodes_ex = None

artifacts = ["MNIST"]  # ["MNIST", "FashionMNIST", "CIFAR10"]

networks = {"NetS": [128] * 3, "NetM": [1024] * 3, "NetL": [1024] * 6}

seeds = [*range(3)]

heuristics = {
    "baseline": None,
    "BS_SDD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "occurrence": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"SDD": {}},
        },
    },
    "BS_SAD": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "occurrence": 5e-3,
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
            "occurrence": 5e-3,
            "start": 1,
            "end": 20,
            "stable_estimator": {"NIP": {"mode": "standard", "epsilon": 0.1}},
        },
    },
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
}

heuristics = {
    "BS_SIP": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 5e-2,
            "occurrence": 5e-3,
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
}

props = [*range(5)]

epsilons = np.linspace(1, 10, 10) / 100

verifiers = ["DNNVWB:neurify", "DNNV:eran_deepzono", "DNNV:nnenum"]
#verifiers = ["DNNVWB:neurify"]
