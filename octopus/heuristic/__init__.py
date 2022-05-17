import torch

from abc import ABC


class Heuristic(ABC):
    def __init__(self, logger):
        self.logger = logger

    def run(self, **kwargs) -> type:
        bool: ...
