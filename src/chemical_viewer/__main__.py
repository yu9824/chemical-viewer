import argparse
import os
import sys
from typing import Optional, Union

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem

from . import __version__
from .viewer import InteractiveChemicalViewer

__all__ = ["main"]


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
        "-f",
        "--file",
        help="file path. columns must have ('x', 'y', 'z', 'texts', 'smiles').",
    )
    args = parser.parse_args(cli_args)

    filepath: Union[os.PathLike, str] = args.file

    if os.path.isfile(filepath):
        df = pd.read_csv(filepath)
        assert {"x", "y", "z", "texts", "smiles"} <= set(
            map(lambda x: x.lower(), df.columns)
        )
    else:
        raise FileNotFoundError(filepath)

    viewer = InteractiveChemicalViewer(width=1)
    mappable = viewer.scatter(
        x=df["x"],
        y=df["y"],
        mols=tuple(map(Chem.MolFromSmiles, df["smiles"])),
        # texts=[f"{i}\n{_smi}" for i, _smi in enumerate(smiles)],
        texts=df["texts"],
        c=df["z"],
    )

    _xlim = viewer.ax.get_xlim()
    _ylim = viewer.ax.get_ylim()
    _lim = min(min(_xlim), min(_ylim)), max(max(_xlim), max(_ylim))
    viewer.ax.set_xlim(_lim)
    viewer.ax.set_ylim(_lim)
    viewer.ax.set_aspect("equal")
    viewer.fig.colorbar(mappable, ax=viewer.ax, label="log S")
    viewer.fig.tight_layout()
    # viewer.scatter(mols, X_umap[:, 0], X_umap[:, 1])
    plt.show()


def entrypoint() -> None:
    main(sys.argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:], prog="chemical-viewer")
