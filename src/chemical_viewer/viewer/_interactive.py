import sys
from copy import copy
from operator import add, attrgetter
from typing import Any, Literal, Optional

import matplotlib.artist
import matplotlib.axes
import matplotlib.backend_bases
import matplotlib.collections
import matplotlib.figure
import matplotlib.offsetbox
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem
from rdkit.Chem import Draw

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


class OffsetImageWithAnnotation(matplotlib.offsetbox.VPacker):
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

    def set_data(
        self,
        arr: ArrayLike,
    ) -> None:
        self.get_children()[0].set_data(arr)

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


def get_xybox(
    xy: tuple[float, float], ax: matplotlib.axes.Axes, alpha=0.25
) -> tuple[float, float]:
    assert len(xy) == 2

    _tup_lim = (ax.get_xlim(), ax.get_ylim())
    _tup_len = tuple(map(lambda _lim: _lim[1] - _lim[0], _tup_lim))
    _xy_center = tuple(map(lambda _lim: add(*_lim) / 2, _tup_lim))

    signs: list[Literal[1, -1]] = list()
    for _t, _t_center in zip(xy, _xy_center):
        signs.append(1 if _t < _t_center else -1)

    return tuple(
        xy[_idx] + _tup_len[_idx] * alpha * signs[_idx] for _idx in range(2)
    )


class InteractiveChemicalViewer:
    def __init__(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        scale: float = 0.3,
    ) -> None:
        self.scale = scale

        # CONSTANTS
        self.LINEWIDTH_INACTIVE = 0.5
        self.LINEWIDTH_ACTIVE = 1

        # ax = plt.gca() if ax is None else ax
        # if is_plotted(ax=ax):
        #     self.ax = ax
        # else:
        #     _, ax = plt.subplots(facecolor="white")
        #     self.ax = ax
        # self.fig = self.ax.figure
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.fig = self.ax.figure

        self.lst_annotations_visible: list[
            matplotlib.offsetbox.AnnotationBbox
        ] = []
        self.annotation_active: Optional[
            matplotlib.offsetbox.AnnotationBbox
        ] = None
        self.annotation_dragging: Optional[
            matplotlib.offsetbox.AnnotationBbox
        ] = None

        self.zorder_max = 3

        # hover時に表示するannotation
        # self.imagebox_hover = matplotlib.offsetbox.OffsetImage(
        #     np.zeros((2, 2)), zoom=self.scale
        # )  # 適当な画像
        # self.imagebox_hover.image.axes = self.ax
        self.imagebox_hover = OffsetImageWithAnnotation(
            np.zeros((2, 2)),
            zoom=self.scale,
            text=None,
        )
        self.imagebox_hover.axes = self.ax

        xy = (add(*self.ax.get_xlim()) / 2, add(*self.ax.get_ylim()) / 2)
        self.annotation_hover = matplotlib.offsetbox.AnnotationBbox(
            self.imagebox_hover,
            xy=xy,
            xybox=xy,
            xycoords="data",
            boxcoords="data",
            pad=0.5,
            arrowprops=dict(arrowstyle="->"),
            zorder=self.zorder_max,
        )
        # 隠しておく
        self.annotation_hover.patch.set_linewidth(self.LINEWIDTH_INACTIVE)
        self.annotation_hover.set_visible(False)
        self.ax.add_artist(self.annotation_hover)

        # 登録
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.fig.canvas.mpl_connect("key_press_event", self.resize_annotation)
        self.fig.canvas.mpl_connect("key_press_event", self.delete)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    # HACK: molsをOptionalにできるようにしたい。
    def scatter(
        self,
        x: ArrayLike,
        y: ArrayLike,
        mols: Sequence[Chem.rdchem.Mol],
        texts: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> matplotlib.collections.PathCollection:
        self.mols = mols

        if texts is None:
            texts = ("",) * len(self.mols)

        assert (
            len(self.mols) == len(x) == len(y) == len(texts)
        ), "lengths of mols, x, and y must be the same"
        self.texts = texts
        self.scatter_object = self.ax.scatter(x, y, **kwargs)
        return self.scatter_object

    def hover(
        self,
        event: matplotlib.backend_bases.MouseEvent,
    ) -> None:
        visibility = self.annotation_hover.get_visible()
        # マウスが乗っているのが当該axならば
        # かつ、annotationのどれにもマウスが乗っていないならば
        if event.inaxes == self.ax:
            # scatter_objectのcontainsメソッドで、マウスが乗っているかどうかを判定
            contains, details = self.scatter_object.contains(event)
            # マウスが乗っている場合
            if (
                contains
                and self.annotation_active is None
                and not any(
                    _annotation.contains(event)[0]
                    for _annotation in self.lst_annotations_visible
                )
            ):
                details: dict[
                    Literal["ind"], np.ndarray[Any, np.dtype[np.int32]]
                ]
                # annotationを更新する作業
                # details["ind"]でマウスが乗っているscatterのindexが取得できる
                # scatterのindexが1次元のnp.ndarrayで返ってくる。（複数が重なっていることもあるため）
                _index_scatter: int = details["ind"][0]

                # 当該点の座標を取得して、位置を更新
                self.annotation_hover.xy = self.scatter_object.get_offsets()[
                    _index_scatter
                ]
                self.annotation_hover.xybox = get_xybox(
                    self.annotation_hover.xy, ax=self.ax, alpha=0.25
                )

                # 画像を更新
                self.imagebox_hover.set_data(
                    Draw.MolToImage(self.mols[_index_scatter])
                )
                # HACK: テキストを指定できるようにする
                self.imagebox_hover.set_text(self.texts[_index_scatter])

                # 見えるようにして
                self.annotation_hover.set_visible(True)
                # 一番手前にして
                if self.annotation_hover.zorder < self.zorder_max:
                    self.zorder_max += 1
                    self.annotation_hover.set(zorder=self.zorder_max)
                # 再描画
                self.fig.canvas.draw_idle()
            # マウスが乗っていない場合
            else:
                # 今annotationが見えているならば
                if visibility:
                    # annotationを見えなくする
                    self.annotation_hover.set_visible(False)
                    # 再描画
                    self.fig.canvas.draw_idle()

    def on_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """クリック時の挙動。

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            event
        """
        # マウスが乗っているのが当該axならば
        if event.inaxes == self.ax:
            # すべてのvisibleなannotationについて
            # for _annotation in self.lst_annotations_visible:
            # zorderの上からそのannotationを触っているのか判定する
            for _annotation in sorted(
                self.lst_annotations_visible, key=attrgetter("zorder")
            )[::-1]:
                contains, _ = _annotation.contains(event)
                # そのannotationにマウスが乗っているならば
                if contains:
                    # 今アクティブなものを非アクティブにする
                    if (
                        type(self.annotation_active)
                        is matplotlib.offsetbox.AnnotationBbox
                    ):
                        self.annotation_active.patch.set_linewidth(
                            self.LINEWIDTH_INACTIVE
                        )
                    # アクティブにする
                    self.annotation_active = _annotation
                    self.annotation_dragging = _annotation
                    # アクティブであることを強調するため、枠線を太くする
                    self.annotation_active.patch.set_linewidth(
                        self.LINEWIDTH_ACTIVE
                    )

                    # 一番手前に持ってくる
                    if _annotation.zorder < self.zorder_max:
                        self.zorder_max += 1
                        _annotation.set(zorder=self.zorder_max)

                    # 次のannotationは見ずに終了
                    break
            # annotationにマウスが乗っていないならば
            else:
                # アクティブなannotationがあるならばそれはとりあえずそれは非アクティブ化する
                if (
                    type(self.annotation_active)
                    is matplotlib.offsetbox.AnnotationBbox
                ):
                    self.annotation_active.patch.set_linewidth(
                        self.LINEWIDTH_INACTIVE
                    )

                # scatter_objectにマウスが乗っているかどうかを判定
                contains, details = self.scatter_object.contains(event)
                # マウスが乗っている場合
                if contains and self.annotation_active is None:
                    index: int = details["ind"][0]

                    _imagebox = OffsetImageWithAnnotation(
                        Draw.MolToImage(self.mols[index]),
                        zoom=self.scale,
                        text=self.texts[
                            index
                        ],  # HACK: テキストを指定できるようにする
                    )
                    _imagebox.axes = self.ax

                    # annotationを作成
                    self.zorder_max += 1
                    xy = self.scatter_object.get_offsets()[index]
                    _annotation = matplotlib.offsetbox.AnnotationBbox(
                        _imagebox,
                        xy=xy,
                        xybox=get_xybox(xy, ax=self.ax, alpha=0.25),
                        xycoords="data",
                        boxcoords="data",
                        # boxcoords="offset points",
                        pad=0.5,
                        arrowprops=dict(arrowstyle="->", facecolor="black"),
                        zorder=self.zorder_max,
                    )
                    _annotation.set_visible(True)
                    self.ax.add_artist(_annotation)

                    # アクティブ化
                    self.annotation_active = _annotation
                    self.annotation_active.patch.set_linewidth(
                        self.LINEWIDTH_ACTIVE
                    )
                    # 移動可能にするために保存
                    self.annotation_dragging = _annotation
                    self.lst_annotations_visible.append(_annotation)
                # 点とannotationのいずれにもマウスが乗っていない場合
                else:
                    # アクティブなannotationがあるならば非アクティブ化する
                    self.annotation_active = None
            self.fig.canvas.draw_idle()

    def on_motion(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """マウスの動きに従ってAnnotationBboxを移動させる。

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            event
        """
        if (
            type(self.annotation_dragging)
            is matplotlib.offsetbox.AnnotationBbox
        ):
            self.annotation_dragging.xybox = (
                event.xdata,
                event.ydata,
            )
            # 再描画
            self.ax.figure.canvas.draw_idle()

    def on_release(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        self.annotation_dragging = None

    def resize_annotation(
        self, event: matplotlib.backend_bases.KeyEvent
    ) -> None:
        """+で拡大。-で縮小。

        もし'active'なannotationがあれば、それだけを拡大する。
        もし'active'なものがないならば全部を拡大する。

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            event

        Raises
        ------
        ValueError
        """
        if event.inaxes == self.ax:
            if (
                type(self.annotation_active)
                is matplotlib.offsetbox.AnnotationBbox
            ):
                offset_image: OffsetImageWithAnnotation = (
                    self.annotation_active.offsetbox
                )
                # *キー/-キーでoffsetboxを拡大縮小できる
                if event.key == "+":
                    offset_image.set_zoom(offset_image.get_zoom() * 1.1)
                elif event.key == "-":
                    offset_image.set_zoom(offset_image.get_zoom() / 1.1)
                # 再描画
                self.fig.canvas.draw_idle()
            elif self.annotation_active is None:
                for _annotation in self.lst_annotations_visible:
                    offset_image: matplotlib.offsetbox.OffsetImage = (
                        _annotation.offsetbox
                    )
                    if event.key == "+":
                        offset_image.set_zoom(offset_image.get_zoom() * 1.1)
                    elif event.key == "-":
                        offset_image.set_zoom(offset_image.get_zoom() / 1.1)
            else:
                raise ValueError
            # 再描画
            self.fig.canvas.draw_idle()

    def delete(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """backspace/deleteで削除。

        もし'active'なannotationがあれば、それだけを削除する。
        もし'active'なものがないならば全部を削除する。

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            event

        Raises
        ------
        ValueError

        """
        if event.inaxes == self.ax:
            if (
                type(self.annotation_active)
                is matplotlib.offsetbox.AnnotationBbox
            ):
                if event.key in {"backspace", "delete"}:
                    self.lst_annotations_visible.remove(self.annotation_active)
                    self.annotation_active.remove()
                    self.annotation_active = None
                    self.annotation_dragging = None
            elif self.annotation_active is None:
                for _annotation in copy(self.lst_annotations_visible):
                    if event.key in {"backspace", "delete"}:
                        self.lst_annotations_visible.remove(_annotation)
                        _annotation.remove()
                        _annotation = None
            else:
                raise ValueError
            # 再描画
            self.fig.canvas.draw_idle()

    # def on_click_point(self, event: matplotlib.backend_bases.MouseEvent) -> None:
    #     if event.inaxes != self.ax:
    #         return

    #     if event.button == 1:
    #         self.add_annotation(event)
