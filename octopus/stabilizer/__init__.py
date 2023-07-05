from abc import ABC

from ..stability_estimator import get_stability_estimators


class Stabilizer(ABC):
    def __init__(self, model, cfg_stable_estimators):
        self.model = model
        self.logger = model.logger
        self._set_stable_estimators(cfg_stable_estimators)

    def run(self, **kwargs) -> type:
        bool: ...

    def _set_stable_estimators(self, cfg_stable_estimators):
        # TODO: only support 1 stable estimator per stabilizer
        stable_estimators = get_stability_estimators(cfg_stable_estimators, self.model)
        if len(stable_estimators) != 1:
            raise ValueError("Only 1 stable estimator is supported for {self}.")
        self.stable_estimators = list(stable_estimators.values())[0]
