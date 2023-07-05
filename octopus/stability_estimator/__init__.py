from .ReLU_estimator.SAD_estimator import SADEstimator
from .ReLU_estimator.SDD_estimator import SDDEstimator
from .ReLU_estimator.NIP_estimator import NIPEstimator
from .ReLU_estimator.SIP_estimator import SIPEstimator
from .ReLU_estimator.ALR_estimator import ALREstimator


def get_stability_estimators(cfg_stable_estimators, model):
    estimators = {}

    for est in cfg_stable_estimators:
        if est in ["SDD", "SAD", "NIP", "SIP", "ALR"]:
            estimator = eval(f"{est}Estimator")(model, **cfg_stable_estimators[est])
        else:
            raise NotImplementedError
        estimators[est] = estimator

    return estimators
