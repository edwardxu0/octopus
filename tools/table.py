import pandas as pd


import argparse


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("T", type=str)

    parser.add_argument("feathers", metavar="N", type=str, nargs="+")

    args = parser.parse_args()
    return args


def main(args):
    if args.T == "T":
        train_table(args)
    elif args.T == "V":
        veri_table(args)


def train_table(args):
    ...


def veri_table(args):
    feathers = pd.read_feather(args.res)
    


if __name__ == "__main__":
    args = _parse_args()
    main(args)
