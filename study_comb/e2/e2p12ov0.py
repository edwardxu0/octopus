from study_comb import *

import pickle
import pathlib

from study_comb import *

artifacts = ["CIFAR10"]  # ["MNIST", "FashionMNIST", "CIFAR10"]
networks = {"OVAL21_o": [1]}

heu_path = pathlib.Path(__file__).resolve()
heu_path = str(heu_path)[:-3] + ".pkl"
with open(heu_path, "rb") as f:
    heuristics = pickle.load(f)
