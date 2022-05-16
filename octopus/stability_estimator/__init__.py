from .ReLU_estimator.SAD_estimator import SADEstimator
from .ReLU_estimator.SDD_estimator import SDDEstimator
from .ReLU_estimator.NIP_estimator import NIPEstimator


def get_stability_estimators(cfg_stable_estimator, model):
    estimators = []

    for est in cfg_stable_estimator:
        if est == "ReLU_estimator":
            estimator = eval(f'{cfg_stable_estimator[est]["mode"]}Estimator')(
                model, **cfg_stable_estimator[est]
            )
        else:
            assert False
        estimators += [estimator]

    return estimators
