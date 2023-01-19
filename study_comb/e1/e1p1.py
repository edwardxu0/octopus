import pickle
import pathlib

from study_comb import *

artifacts = ["MNIST"]  # ["MNIST", "FashionMNIST", "CIFAR10"]
networks = {"NetS": [128] * 3}

heu_path = pathlib.Path(__file__).resolve()
heu_path = str(heu_path)[:-3] + ".pkl"
with open(heu_path, "rb") as f:
    heuristics = pickle.load(f)
