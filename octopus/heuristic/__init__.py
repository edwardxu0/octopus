import torch

from abc import ABC


class Heuristic(ABC):
    def __init__(self):
        ...

    def run(self, type: torch.tensor) -> type: bool: ...
