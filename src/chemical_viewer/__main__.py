import argparse
import sys
from typing import Optional

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

from . import __version__
from .gui import main_tk

__all__ = ("main",)


# HACK: argumentに--fileを加える。テンプレートのCSVを用意してそれを読み込む方式。
def main(cli_args: Sequence[str], prog: Optional[str] = None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description="")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        help="show current version",
        version=f"%(prog)s: {__version__}",
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="file path. columns must have 'x', 'y', 'smiles'. ('z', 'texts' are optional)",
    )
    args = parser.parse_args(cli_args)

    main_tk(args.file)


def entrypoint() -> None:
    main(sys.argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:], prog="chemical-viewer")
