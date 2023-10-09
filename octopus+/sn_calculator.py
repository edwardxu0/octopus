# TODO: use this as a function...

from pathlib import Path
import torch.nn as nn
import random
import torch
import time
import os
import sys

from verifier.verifier import Verifier
from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx
from verifier.objective import Objective, DnfObjectives
from util.misc.logger import logger
from setting import Settings


def extract_instance(net_path, vnnlib_path):
    vnnlibs = read_vnnlib(Path(vnnlib_path))
    model, input_shape, output_shape, is_nhwc = parse_onnx(net_path)

    # objective
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
    objectives = DnfObjectives(objectives, input_shape=input_shape, is_nhwc=is_nhwc)

    return model, input_shape, objectives


def calculate_meta(net_path, vnnlib_path):
    device = "cpu"

    # print("Running test with", net_path, vnnlib_path)
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)

    verifier = Verifier(
        net=model,
        input_shape=input_shape,
        batch=1000,
        device=device,
    )

    stable, unstable, lbs, ubs = verifier.compute_stability(objectives)
    # print("# stable:", stable.numpy())
    # print("# unstable:", unstable.numpy())
    # print("lbs:", lbs)
    # print("ubs:", ubs)

    return int(stable.numpy()), int(unstable.numpy())
