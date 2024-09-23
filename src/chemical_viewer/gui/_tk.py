import os
import tkinter
from pathlib import Path
from typing import Optional, Union

import matplotlib.axes
import matplotlib.figure
import pandas as pd
import TkEasyGUI as eg
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from rdkit import Chem

from chemical_viewer.logging import DEBUG, get_child_logger
from chemical_viewer.viewer import InteractiveChemicalViewer

_logger = get_child_logger(__name__)


def _draw_figure(
    figure: matplotlib.figure.Figure,
    canvas: Union[tkinter.Canvas, eg.Canvas],
) -> FigureCanvasTkAgg:
    figure_canvas = FigureCanvasTkAgg(figure, canvas)
    NavigationToolbar2Tk(figure_canvas)
    figure_canvas.draw()
    figure_canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas


def plot(df: pd.DataFrame, values: dict) -> None:
    viewer = InteractiveChemicalViewer()
    viewer.scatter(
        x=df[values["x"]],
        y=df[values["y"]],
        c=df[values["z"]] if values["z"] else None,
        mols=tuple(map(Chem.MolFromSmiles, df[values["smiles"]])),
        texts=df[values["texts"]] if values["texts"] else None,
    )
    viewer.ax.set_aspect(1.0 / viewer.ax.get_data_ratio(), adjustable="box")

    with eg.Window(
        title="main",
        layout=((eg.Canvas(key="-CANVAS-", size=(200, 200)),),),
        resizable=True,
        keep_on_top=False,
    ) as window:
        _draw_figure(figure=viewer.fig, canvas=window["-CANVAS-"])
        for event, values in window.event_iter():
            if event in {eg.WINDOW_CLOSED}:
                break
            elif event == "Reload":
                viewer.ax.set_xlim(
                    *tuple(float(values[f"xlim{_idx}"]) for _idx in range(2))
                )
                viewer.ax.set_ylim(
                    *tuple(float(values[f"ylim{_idx}"]) for _idx in range(2))
                )

            viewer.ax.set_aspect(
                1.0 / viewer.ax.get_data_ratio(), adjustable="box"
            )


def main_tk(filepath_input: Optional[Union[os.PathLike, str]]) -> None:
    if not filepath_input:
        filepath_input = eg.popup_get_file(
            message="select data file",
            file_types=(("CSV", "*.csv"), ("Excel", "*.xlsx")),
            multiple_files=False,
        )
    filepath_input = Path(filepath_input)

    if filepath_input.is_file():
        if filepath_input.suffix == ".csv":
            df = pd.read_csv(filepath_input)
        elif filepath_input.suffix == ".xlsx":
            df = pd.read_excel(".xlsx")
        else:
            raise ValueError
    else:
        raise FileNotFoundError

    start(df)


def start(df: pd.DataFrame) -> None:
    layout = (
        (
            eg.Table(
                values=df.head().values.tolist(),
                headings=df.columns.tolist(),
                key="table",
                max_col_width=1,
                expand_x=True,
            ),
        ),
        (
            eg.Text("x"),
            eg.Combo(values=df.columns.tolist(), key="x", default_value="x"),
        ),
        (
            eg.Text("y"),
            eg.Combo(values=df.columns.tolist(), key="y", default_value="y"),
        ),
        (
            eg.Text("z (optional)"),
            eg.Combo(values=df.columns.tolist(), key="z", default_value=""),
        ),
        (
            eg.Text("smiles"),
            eg.Combo(
                values=df.columns.tolist(),
                key="smiles",
                default_value="smiles",
            ),
        ),
        (
            eg.Text("texts (optional)"),
            eg.Combo(
                values=df.columns.tolist(), key="texts", default_value=""
            ),
        ),
        (eg.Button("OK"), eg.Button("Cancel")),
    )

    with eg.Window(
        title="Start Menu",
        layout=layout,
        resizable=True,
        size=(600, 600),
    ) as window:
        for event, values in window.event_iter():
            if event in {eg.WINDOW_CLOSED, "Cancel"}:
                break
            elif event == "OK":
                plot(df, values)


if __name__ == "__main__":
    _logger.setLevel(DEBUG)
    main_tk()
