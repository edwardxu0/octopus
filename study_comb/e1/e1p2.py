import pickle
import pathlib

from study_comb import *

artifacts = ["FashionMNIST"]  # ["MNIST", "FashionMNIST", "CIFAR10"]
networks = {"NetM": [1024] * 3}

heu_path = pathlib.Path(__file__).resolve()
heu_path = str(heu_path)[:-3] + ".pkl"
with open(heu_path, "rb") as f:
    heuristics = pickle.load(f)
