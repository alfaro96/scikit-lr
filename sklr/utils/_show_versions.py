"""Print the information needed to debug a problem with :mod:`sklr`."""

import sklearn

from sklr import __version__
from sklr._build_info import SKLEARN_BUILD_VERSION


def show_versions() -> None:
    """Print useful debugging information.

    Print the version of scikit-lr and the one of scikit-learn that it was built
    against, followed by the information of :func:`sklearn.show_versions`: the
    system, the versions of the main dependencies and the threading libraries.

    See Also
    --------
    sklearn.show_versions : Print the debugging information of scikit-learn.

    Examples
    --------
    >>> from sklr import show_versions
    >>> show_versions()  # doctest: +SKIP
    """
    print("\nscikit-lr:")
    print(f"{'sklr':>13}: {__version__}")
    # The extension modules only work with the minor release of scikit-learn
    # that they were built against, so it is reported along with the one that
    # sklearn.show_versions finds installed
    print(f"{'built with':>13}: scikit-learn {SKLEARN_BUILD_VERSION}")
    sklearn.show_versions()
