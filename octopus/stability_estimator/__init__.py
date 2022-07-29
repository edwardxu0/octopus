from .ReLU_estimator.SAD_estimator import SADEstimator
from .ReLU_estimator.SDD_estimator import SDDEstimator
from .ReLU_estimator.NIP_estimator import NIPEstimator
from .ReLU_estimator.SIP_estimator import SIPEstimator


def get_stability_estimators(cfg_stable_estimator, model):
    estimators = {}

    for est in cfg_stable_estimator:
        if est in ["SDD", "SAD", "NIP", "SIP"]:
            estimator = eval(f"{est}Estimator")(model, **cfg_stable_estimator[est])
        else:
            raise NotImplementedError
        estimators[est] = estimator

    return estimators
