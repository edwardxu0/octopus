import argparse
import logging

from pyfiglet import Figlet
from datetime import datetime

from octopus.misc.logging import initialize_logger

from .benchmark import Benchmark


def _parse_args():
    parser = argparse.ArgumentParser(description="OCTOPUS Benchmarks.", prog="octopus+")
    parser.add_argument(
        "base_settings", type=str, help="An OCTOPUS toml config file as bases."
    )
    parser.add_argument(
        "benchmark_settings",
        type=str,
        help="A python file containing the parameters to explore.",
    )
    parser.add_argument(
        "task",
        type=str,
        choices=["T",  "V", "A"],
        help="Tasks to perform on the benchmark, including [T]rain, [V]erify, and [A]nalyze.",
    )
    parser.add_argument(
        "--result_dir", type=str, default="./results", help="Root result directory."
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Overrides existing train/verify/analyze tasks.",
    )
    parser.add_argument("--go", action="store_true", help="Dry run or not?")
    parser.add_argument("--debug", action="store_true", help="Print debug log.")
    parser.add_argument("--dumb", action="store_true", help="Silent mode.")
    return parser.parse_args()


def main():
    f = Figlet(font="slant")
    print(f.renderText("OCTOPUS +"), end="")

    time_start = datetime.now()
    args = _parse_args()

    if args.debug:
        logging_level = logging.DEBUG
    elif args.dumb:
        logging_level = logging.WARN
    else:
        logging_level = logging.INFO

    logger = initialize_logger("OCTOPUS+", logging_level)
    study = Benchmark(
        args.base_settings,
        args.benchmark_settings,
        go=args.go,
        override=args.override,
        result_dir=args.result_dir,
        logger=logger,
    )

    if args.task == "T":
        study.train()

    elif args.task == "Test":
        study.test()

    elif args.task == "V":
        study.verify()

    elif args.task == "A":
        study.analyze()

    else:
        assert False

    logger.info("Mission Complete.")
    time_end = datetime.now()
    duration = time_end - time_start
    logger.info(f"Spent {duration.seconds} seconds.")


if __name__ == "__main__":
    main()
