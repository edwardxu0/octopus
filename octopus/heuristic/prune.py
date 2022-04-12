from . import Heuristic

class Prune(Heuristic):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

        # restructure architecture to remove empty neurons
    def restructure(self):
        pass

    # pruning code ... 
    def run(self, cfg):
        pass
