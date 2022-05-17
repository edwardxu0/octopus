import sys
import logging

logging.basicConfig(
    stream=sys.stdout,
    format="[%(name)s](%(levelname)s) %(asctime)s -> %(message)s ",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def initialize_logger(name, logging_level):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging_level)
    return logger
