from .misc import cli
from .misc import config
from .misc import logging
from .core import workout


def core():
    args = cli.parse_args()
    settings = config.configure(args)
    logging.initialize(settings)
    
    workout(settings)

if __name__ == "__main__":
    core()
