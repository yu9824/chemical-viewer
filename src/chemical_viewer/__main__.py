import argparse
import sys
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

from chemical_viewer import __version__
from chemical_viewer.gui import main_tk
from chemical_viewer.logging import DEBUG, get_child_logger, get_root_logger

__all__ = ("main",)

root_logger = get_root_logger()
_logger = get_child_logger(__name__)


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
        "-d", "--debug", action="store_true", help="run in debug mode"
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=None,
        help="file path. columns must have 'x', 'y', 'smiles'. ('z', 'texts' are optional)",
    )
    args = parser.parse_args(cli_args)

    if args.debug:
        root_logger.setLevel(DEBUG)

    try:
        main_tk(args.file)
    except Exception:
        _logger.exception("")


def entrypoint() -> None:
    main(sys.argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:], prog="chemical-viewer")
