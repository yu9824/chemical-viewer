"""utils module"""

from ._utils import dummy_func, is_installed, is_plotted

# _utils.pyだと、_が入っているのでドキュメント化されない。
# ドキュメント化したい場合は、モジュールメソッドとして登録するため、__all__に入れる。
__all__ = ("is_installed", "dummy_func", "is_plotted")
