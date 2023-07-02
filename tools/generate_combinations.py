#!/usr/bin/env python
import copy
import itertools
import sys
import pickle

stabilizer_skeleton = {
    "B": {
        "bias_shaping": {
            "mode": "standard",
            "intensity": 1e-2,
            "pace": 100,
            "start": [],
            "end": [],
        }
    },
    "R": {
        "rs_loss": {
            "mode": "standard",
            "weight": 1e-4,
            "start": [],
            "end": [],
        }
    },
    "P": {
        "prune": {
            "mode": "stablenet",
            "sparsity": 1e-2,
            "pace": 100,
            "start": [],
            "end": [],
        }
    },
}

estimator_skeleton = {"SDD": {"SDD": {}},
                      "SIP": {"SIP": {"mode": "standard", "epsilon": 0.1}}}


def main(args):
    steps = 3
    stride = 7
    #estimator = "SDD"
    estimator = "SIP"
    stabilizers = ["B", "R", "P", "N"]

    heuristics = ["".join(x) for x in itertools.product(stabilizers, repeat=steps)]

    H = {}
    for name in heuristics:
        heu = {}
        heu_shortcuts = {}
        for i, s in enumerate(name):
            # print(i, s)
            if s == "N":
                continue
            start = i * stride + 1
            end = (i + 1) * stride

            if not s in heu_shortcuts:
                stabilizer = copy.deepcopy(stabilizer_skeleton[s])
                assert len(stabilizer.keys()) == 1
                s_ = list(stabilizer.keys())[0]

                stabilizer = stabilizer[s_]
                stabilizer["stable_estimator"] = estimator_skeleton[estimator]
                heu[s_] = stabilizer
                heu_shortcuts[s] = s_
            else:
                stabilizer = heu[heu_shortcuts[s]]

            stabilizer["start"] += [start]
            stabilizer["end"] += [end]

        H[name] = heu

    save_path = args[1]
    print(f"Saving heuristics to {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(H, f)


if __name__ == "__main__":
    main(sys.argv)
