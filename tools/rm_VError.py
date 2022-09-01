#!/usr/bin/env python

import os
import sys
import importlib
import toml


def main():
    print("Usage: ./rm_VError.py [octopus+ .toml] [octopus+ .py]")

    ROOT = "./results"

    verifier_encoder = {
        "eran_deepzono": "ERAN",
        "eran_deeppoly": "ERAN",
        "eran_refinezono": "ERAN",
        "eran_refinepoly": "ERAN",
        "neurify": "Neurify",
        "nnenum": "Nnenum",
        "marabou": "Marabou",
        "planet": "Planet",
    }

    toml_config = toml.load(open(sys.argv[1], "r"))
    study_path = sys.argv[2].split(".")[0].replace("/", ".")
    module = importlib.import_module(study_path)

    name = toml_config["name"]
    verifiers = module.__dict__["verifiers"]
    print("Verifiers: ", verifiers)

    veri_log_dir = os.path.join(ROOT, name, "veri_log")
    files = os.listdir(veri_log_dir)

    for f in files:
        for v in verifiers:
            v = v.split(":")[1]
            log_path = os.path.join(ROOT, name, "veri_log", f)
            lines = open(log_path, "r").readlines()
            lines = [x for x in lines if "result:" in x or "slurmstepd" in x]
            if len(lines) == 2:
                lines = lines[1:]
            bad = False
            tag = [f"result: {verifier_encoder[v]}Error(Return code",
                   "CANCELLED"]

            for l in lines:
                if [True for x in tag if x in l]:
                    bad = True
                    break
                
            if bad:
                print(v, log_path)
                if len(sys.argv) == 4 and sys.argv[3] == "--go":
                    cmd = f"rm {log_path}"
                    os.system(cmd)
                break


if __name__ == "__main__":
    main()
