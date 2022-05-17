from pyfiglet import Figlet

from .misc import cli
from .misc import config
from .core import workout


def main():
    f = Figlet(font="slant")
    print(f.renderText("OCTOPUS"), end="")

    args = cli.parse_args()
    settings = config.configure(args)

    workout(settings)


if __name__ == "__main__":
    main()
