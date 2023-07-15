import os
import toml
import pathlib
import logging

from .logging import initialize_logger


def configure(args):
    with open(args.configs, "r") as config_file:
        cfg = toml.loads(config_file.read())
    cfg["result_dir"] = os.path.join(args.result_dir, f'{cfg["name"]}')
    cfg["seed"] = args.seed
    cfg["task"] = args.task
    cfg["override"] = args.override

    if args.debug:
        cfg["logging_level"] = logging.DEBUG
    elif args.dumb:
        cfg["logging_level"] = logging.WARN
    else:
        cfg["logging_level"] = logging.INFO

    settings = Settings(cfg)
    return settings


class Settings:
    def __init__(self, cfg):
        self.name = cfg["name"]
        self.seed = cfg["seed"]
        self.task = cfg["task"]
        self.override = cfg["override"]
        self.result_dir = cfg["result_dir"]
        self.sub_dirs = {"result_dir": self.result_dir}

        self.cfg_train = cfg["train"]
        self.cfg_stabilizers = cfg["stabilizers"] if "stabilizers" in cfg else None
        self.cfg_verify = cfg["verify"]

        self.answer_code = {
            "unsat": 1,
            "sat": 2,
            "unknown": 3,
            "timeout": 4,
            "memout": 4,
            "error": 5,
        }

        sub_dirs = {"train_log", "model", "property", "figure", "veri_log"}
        self._make_dirs(sub_dirs)
        self.logger = initialize_logger("OCTOPUS", cfg["logging_level"])

    def _make_dirs(self, sub_dirs):
        pathlib.Path().mkdir(parents=True, exist_ok=True)
        for sd in sub_dirs:
            sdp = os.path.join(self.result_dir, sd)
            self.sub_dirs[sd + "_dir"] = sdp
            pathlib.Path(sdp).mkdir(parents=True, exist_ok=True)
