import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Verifiable Networks", prog="octopus"
    )

    parser.add_argument("configs", type=str, help="Configurations file.")
    parser.add_argument(
        "task",
        type=str,
        choices=["T", "Test", "V", "A", "AA"],
        help="Select tasks to perform, including [T]rain, [V]erify, [A]nalyze, [AA]ll 'bove.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--result_dir", type=str, default="./results/", help="Result directory."
    )
    parser.add_argument(
        "--override", action="store_true", help="Overrides training/verification tasks."
    )
    parser.add_argument("--debug", action="store_true", help="Print debug log.")
    parser.add_argument("--dumb", action="store_true", help="Silent mode.")
    parser.add_argument("--version", action="version", version="%(prog)s 0.3.1")
    return parser.parse_args()
