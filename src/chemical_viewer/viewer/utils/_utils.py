import sys
from operator import add
from typing import Any, Optional

import matplotlib.offsetbox
import numpy as np
from numpy.typing import ArrayLike

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class OffsetImageWithAnnotation(matplotlib.offsetbox.VPacker):
    """_summary_

    Parameters
    ----------
    arr : ArrayLike
        image array
    zoom : float, optional
        zoom scale, by default 0.3
    text : Optional[str], optional
        text, by default None
    """

    def __init__(
        self,
        arr: ArrayLike,
        zoom: float = 0.3,
        text: Optional[str] = None,
    ) -> None:
        children = [
            matplotlib.offsetbox.OffsetImage(arr, zoom=zoom),
            matplotlib.offsetbox.TextArea(text if text else ""),
        ]
        super().__init__(children=children, align="left")

    def get_children(
        self,
    ) -> tuple[
        matplotlib.offsetbox.OffsetImage, matplotlib.offsetbox.TextArea
    ]:
        return tuple(super().get_children())

    def get_data(
        self,
    ) -> "np.ndarray[Any, int]":
        return self.get_children()[0].get_data()

    def set_data(
        self,
        arr: ArrayLike,
    ) -> None:
        self.get_children()[0].set_data(arr)

    def get_text(self) -> str:
        return self.get_children()[1].get_text()

    def set_text(
        self,
        text: str,
    ) -> None:
        self.get_children()[1].set_text(text)

    def get_zoom(
        self,
    ) -> float:
        return self.get_children()[0].get_zoom()

    def set_zoom(
        self,
        zoom: float,
    ) -> None:
        zoom_before = self.get_zoom()

        text_object: matplotlib.text.Text = self.get_children()[
            1
        ].get_children()[0]
        text_object.set_fontsize(
            zoom / zoom_before * text_object.get_fontsize()
        )
        self.get_children()[0].set_zoom(zoom)

    def is_same(self, offsetbox: "OffsetImageWithAnnotation") -> bool:
        return (
            np.allclose(offsetbox.get_data(), self.get_data())
            and offsetbox.get_text() == self.get_text()
        )


def get_xybox(
    xy: tuple[float, float], ax: matplotlib.axes.Axes, alpha: float = 0.25
) -> tuple[float, float]:
    """get xybox coordination from xy coordination

    Parameters
    ----------
    xy : tuple[float, float]
        xy coordination
    ax : matplotlib.axes.Axes
        Axes
    alpha : float, optional
        scale parameter, by default 0.25

    Returns
    -------
    tuple[float, float]
        xybox coordination
    """
    assert len(xy) == 2

    _tup_lim = (ax.get_xlim(), ax.get_ylim())
    _tup_len = tuple(map(lambda _lim: _lim[1] - _lim[0], _tup_lim))
    _xy_center = tuple(map(lambda _lim: add(*_lim) / 2, _tup_lim))

    signs: "tuple[Literal[1, -1], Literal[1, -1]]" = tuple(
        1 if _t < _t_center else -1 for _t, _t_center in zip(xy, _xy_center)
    )

    return tuple(
        xy[_idx] + _tup_len[_idx] * alpha * signs[_idx] for _idx in range(2)
    )
