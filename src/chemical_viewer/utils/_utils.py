import pkgutil
import warnings
from typing import (
    Optional,
    TypeVar,  # deprecated in python >=3.12
)

import matplotlib.axes
import matplotlib.pyplot as plt

T = TypeVar("T")


PACKAGE_NAMES = {_module.name for _module in pkgutil.iter_modules()}


def is_installed(package_name: str) -> bool:
    """Check if the package is installed.

    Parameters
    ----------
    package_name : str
        package name like `sklearn`

    Returns
    -------
    bool
        if installed, True
    """
    return package_name in PACKAGE_NAMES


def dummy_func(x: T, *args, **kwargs) -> T:
    """dummy function

    Parameters
    ----------
    x : T
        Anything

    Returns
    -------
    T
        same as input
    """
    return x


def is_plotted(ax: Optional[matplotlib.axes.Axes] = None) -> bool:
    """Check if an axes has been plotted on.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None
        The axes to check. If None, the current axes will be used.
        optional, by default None

    Returns
    -------
    bool
        True if the axes has been plotted on, False otherwise.
    """
    ax = plt.gca() if ax is None else ax
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)

        return any(
            len(getattr(ax, _key))
            for _key in dir(ax)
            if isinstance(getattr(ax, _key), matplotlib.axes.Axes.ArtistList)
        )
