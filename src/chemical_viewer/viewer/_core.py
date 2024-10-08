import sys
from copy import copy
from operator import add, attrgetter
from typing import Any, Optional

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

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

from chemical_viewer.viewer.utils import OffsetImageWithAnnotation, get_xybox


class InteractiveChemicalViewer:
    """InteractiveChemicalViewer

    Parameters
    ----------
    ax : Optional[matplotlib.axes.Axes], optional
        _description_, by default None
    scale : float, optional
        _description_, by default 0.3

    """

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

        self._annotations_visible: tuple[
            matplotlib.offsetbox.AnnotationBbox, ...
        ] = ()
        self._annotation_active: Optional[
            matplotlib.offsetbox.AnnotationBbox
        ] = None
        self._annotation_dragging: Optional[
            matplotlib.offsetbox.AnnotationBbox
        ] = None

        self.mols: tuple[tuple[Chem.rdchem.Mol, ...]] = ()
        self.texts: tuple[tuple[str, ...]] = ()
        self.scatter_objects: tuple[
            matplotlib.collections.PathCollection,
            ...,
        ] = ()
        # self._indexes_visible: tuple[tuple[int, int], ...] = ()

        self.zorder_max = 3

        # hover時に表示するannotation
        self._imagebox_hover = OffsetImageWithAnnotation(
            np.zeros((2, 2)),
            zoom=self.scale,
            text=None,
        )
        self._imagebox_hover.axes = self.ax

        xy = (add(*self.ax.get_xlim()) / 2, add(*self.ax.get_ylim()) / 2)
        self._annotation_hover = matplotlib.offsetbox.AnnotationBbox(
            self._imagebox_hover,
            xy=xy,
            xybox=xy,
            xycoords="data",
            boxcoords="data",
            pad=0.5,
            arrowprops=dict(arrowstyle="->"),
            zorder=self.zorder_max,
        )
        # 隠しておく
        self._annotation_hover.patch.set_linewidth(self.LINEWIDTH_INACTIVE)
        self._annotation_hover.set_visible(False)
        self.ax.add_artist(self._annotation_hover)

        # 登録
        self.fig.canvas.mpl_connect("motion_notify_event", self._hover)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self.fig.canvas.mpl_connect("key_press_event", self._resize_annotation)
        self.fig.canvas.mpl_connect("key_press_event", self._delete)

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
        """wrapper of `plt.scatter`

        Parameters
        ----------
        x : ArrayLike
            x
        y : ArrayLike
            y
        mols : Sequence[Chem.rdchem.Mol]
            mols
        texts : Optional[Sequence[str]], optional
            texts, by default None

        Returns
        -------
        matplotlib.collections.PathCollection
            return of `plt.scatter`
        """
        self.mols += (tuple(mols),)

        if texts is None:
            texts = ("",) * len(mols)
        else:
            texts = tuple(texts)
        self.texts += (texts,)

        assert (
            len(mols) == len(x) == len(y) == len(texts)
        ), "lengths of mols, x, and y must be the same"

        _scatter_object = self.ax.scatter(x, y, **kwargs)
        self.scatter_objects += (_scatter_object,)
        return _scatter_object

    def _hover(
        self,
        event: matplotlib.backend_bases.MouseEvent,
    ) -> None:
        visibility = self._annotation_hover.get_visible()
        # マウスが乗っているのが当該axならば
        # かつ、annotationのどれにもマウスが乗っていないならば
        if event.inaxes == self.ax:
            # zorderが大きい順に並び替えて、同じ場合は後からscatterされたもの
            for _scatter_object in sorted(
                self.scatter_objects,
                key=lambda _scatter_object: (
                    getattr(_scatter_object, "zorder"),
                    self.scatter_objects.index(_scatter_object),
                ),
                reverse=True,
            ):
                # scatter_objectのcontainsメソッドで、マウスが乗っているかどうかを判定
                contains, details = _scatter_object.contains(event)
                # マウスが乗っている場合
                if (
                    contains
                    and self._annotation_active is None
                    and not any(
                        _annotation.contains(event)[0]
                        for _annotation in self._annotations_visible
                    )
                ):
                    details: dict[
                        Literal["ind"], np.ndarray[Any, np.dtype[np.int32]]
                    ]
                    # annotationを更新する作業
                    # details["ind"]でマウスが乗っているscatterのindexが取得できる
                    # scatterのindexが1次元のnp.ndarrayで返ってくる。（複数が重なっていることもあるため）
                    # _index_marker: int = details["ind"][0]
                    for _index_marker in details["ind"]:
                        _index_scatter_object: int = (
                            self.scatter_objects.index(_scatter_object)
                        )

                        # 当該点の座標を取得して、位置を更新
                        self._annotation_hover.xy = (
                            _scatter_object.get_offsets()[_index_marker]
                        )
                        self._annotation_hover.xybox = get_xybox(
                            self._annotation_hover.xy, ax=self.ax, alpha=0.25
                        )

                        # 色をプロットしている色と同じにする
                        if _scatter_object.get_edgecolor().shape[0] == 1:
                            edgecolor = _scatter_object.get_edgecolor()
                        else:
                            # 連続値の場合は黒
                            edgecolor = "black"
                        self._annotation_hover.patch.set_edgecolor(edgecolor)

                        # 画像を更新
                        self._imagebox_hover.set_data(
                            Draw.MolToImage(
                                self.mols[_index_scatter_object][_index_marker]
                            )
                        )
                        self._imagebox_hover.set_text(
                            self.texts[_index_scatter_object][_index_marker]
                        )

                        # 完全に同じ座標、画像、テキストの場合、スキップ
                        if any(
                            self._imagebox_hover.is_same(_annotation.offsetbox)
                            and tuple(_annotation.xy)
                            == tuple(self._annotation_hover.xy)
                            for _annotation in self._annotations_visible
                        ):
                            continue

                        # 見えるようにして
                        self._annotation_hover.set_visible(True)
                        # 一番手前にして
                        if self._annotation_hover.zorder < self.zorder_max:
                            self.zorder_max += 1
                            self._annotation_hover.set(zorder=self.zorder_max)

                        # 再描画
                        self.fig.canvas.draw_idle()

                        # 描画できたので終了
                        break
                    else:
                        # 描画できなかったので別のscatter_objectを探す
                        continue

                    # 描画できたので終了
                    break
                # マウスが乗っていない場合
                else:
                    # 今annotationが見えているならば
                    if visibility:
                        # annotationを見えなくする
                        self._annotation_hover.set_visible(False)
                        # 再描画
                        self.fig.canvas.draw_idle()

    def _on_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """クリック時の挙動。

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            event
        """
        # マウスが乗っているのが当該axならば
        if event.inaxes == self.ax:
            # すべてのvisibleなannotationについて
            # for _annotation in self._annotations_visible:
            # zorderの上からそのannotationを触っているのか判定する
            for _annotation in sorted(
                self._annotations_visible,
                key=attrgetter("zorder"),
                reverse=True,
            ):
                contains, _ = _annotation.contains(event)
                # そのannotationにマウスが乗っているならば
                if contains:
                    # 今アクティブなものを非アクティブにする
                    if (
                        type(self._annotation_active)
                        is matplotlib.offsetbox.AnnotationBbox
                    ):
                        self._annotation_active.patch.set_linewidth(
                            self.LINEWIDTH_INACTIVE
                        )
                    # アクティブにする
                    self._annotation_active = _annotation
                    self._annotation_dragging = _annotation
                    # アクティブであることを強調するため、枠線を太くする
                    self._annotation_active.patch.set_linewidth(
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
                    type(self._annotation_active)
                    is matplotlib.offsetbox.AnnotationBbox
                ):
                    self._annotation_active.patch.set_linewidth(
                        self.LINEWIDTH_INACTIVE
                    )

                # zorderが大きい順に並び替えて、同じ場合は後からscatterされたもの
                for _scatter_object in sorted(
                    self.scatter_objects,
                    key=lambda _scatter_object: (
                        getattr(_scatter_object, "zorder"),
                        self.scatter_objects.index(_scatter_object),
                    ),
                    reverse=True,
                ):
                    # scatter_objectにマウスが乗っているかどうかを判定
                    contains, details = _scatter_object.contains(event)
                    # マウスが乗っている場合
                    if contains and self._annotation_active is None:
                        # _index_marker: int = details["ind"][0]
                        for _index_marker in details["ind"]:
                            _index_scatter_object: int = (
                                self.scatter_objects.index(_scatter_object)
                            )
                            # self._indexes_visible += (
                            #     (_index_scatter_object, _index_marker),
                            # )

                            _imagebox = OffsetImageWithAnnotation(
                                Draw.MolToImage(
                                    self.mols[_index_scatter_object][
                                        _index_marker
                                    ]
                                ),
                                zoom=self.scale,
                                text=self.texts[_index_scatter_object][
                                    _index_marker
                                ],
                            )
                            _imagebox.axes = self.ax

                            xy = tuple(
                                _scatter_object.get_offsets()[_index_marker]
                            )
                            # 完全に同じ座標、画像、テキストの場合、スキップ
                            if any(
                                _imagebox.is_same(_annotation.offsetbox)
                                and xy == tuple(_annotation.xy)
                                for _annotation in self._annotations_visible
                            ):
                                continue

                            # annotationを作成
                            self.zorder_max += 1
                            _annotation = matplotlib.offsetbox.AnnotationBbox(
                                _imagebox,
                                xy=xy,
                                xybox=get_xybox(xy, ax=self.ax, alpha=0.25),
                                xycoords="data",
                                boxcoords="data",
                                # boxcoords="offset points",
                                pad=0.5,
                                arrowprops=dict(
                                    arrowstyle="->", facecolor="black"
                                ),
                                zorder=self.zorder_max,
                            )
                            _annotation.set_visible(True)

                            # 色をプロットしている色と同じにする
                            if _scatter_object.get_edgecolor().shape[0] == 1:
                                edgecolor = _scatter_object.get_edgecolor()
                            else:
                                # 連続値の場合は黒
                                edgecolor = "black"
                            _annotation.patch.set_edgecolor(edgecolor)

                            self.ax.add_artist(_annotation)

                            # アクティブ化
                            self._annotation_active = _annotation
                            self._annotation_active.patch.set_linewidth(
                                self.LINEWIDTH_ACTIVE
                            )
                            # 移動可能にするために保存
                            self._annotation_dragging = _annotation
                            self._annotations_visible += (_annotation,)

                            # 描画できたので終了
                            break
                        else:
                            # 描画できなかったので別のscatter_objectを探す
                            continue
                        # 描画できたので終了
                        break
                # 点とannotationのいずれにもマウスが乗っていない場合
                else:
                    # アクティブなannotationがあるならば非アクティブ化する
                    self._annotation_active = None
            self.fig.canvas.draw_idle()

    def _on_motion(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """マウスの動きに従ってAnnotationBboxを移動させる。

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            event
        """
        if (
            type(self._annotation_dragging)
            is matplotlib.offsetbox.AnnotationBbox
        ):
            self._annotation_dragging.xybox = (
                event.xdata,
                event.ydata,
            )
            # 再描画
            self.ax.figure.canvas.draw_idle()

    def _on_release(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        self._annotation_dragging = None

    def _resize_annotation(
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
                type(self._annotation_active)
                is matplotlib.offsetbox.AnnotationBbox
            ):
                offset_image: OffsetImageWithAnnotation = (
                    self._annotation_active.offsetbox
                )
                # *キー/-キーでoffsetboxを拡大縮小できる
                if event.key == "+":
                    offset_image.set_zoom(offset_image.get_zoom() * 1.1)
                elif event.key == "-":
                    offset_image.set_zoom(offset_image.get_zoom() / 1.1)
                # 再描画
                self.fig.canvas.draw_idle()
            elif self._annotation_active is None:
                for _annotation in self._annotations_visible:
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

    def _delete(self, event: matplotlib.backend_bases.KeyEvent) -> None:
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
            _list_annotations_visible = list(self._annotations_visible)
            if (
                type(self._annotation_active)
                is matplotlib.offsetbox.AnnotationBbox
            ):
                if event.key in {"backspace", "delete"}:
                    _list_annotations_visible.remove(self._annotation_active)
                    self._annotation_active.remove()
                    self._annotation_active = None
                    self._annotation_dragging = None
            elif self._annotation_active is None:
                for _annotation in copy(self._annotations_visible):
                    if event.key in {"backspace", "delete"}:
                        _list_annotations_visible.remove(_annotation)
                        _annotation.remove()
                        _annotation = None
            else:
                raise ValueError
            self._annotations_visible = tuple(_list_annotations_visible)
            # 再描画
            self.fig.canvas.draw_idle()

    # def on_click_point(self, event: matplotlib.backend_bases.MouseEvent) -> None:
    #     if event.inaxes != self.ax:
    #         return

    #     if event.button == 1:
    #         self.add_annotation(event)
