from . import *

artifacts = ["CIFAR10"]
networks = {"CIFAR2020_8_255": [49402]}

stabilizers = {
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
}
