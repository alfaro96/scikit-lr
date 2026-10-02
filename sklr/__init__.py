"""Label ranking and partial label ranking estimators for scikit-learn."""

import importlib as _importlib

# This line is also parsed by sklr/_build_utils/version.py to set the
# version of the distribution, so keep it as a single string literal.
__version__ = "0.3.dev0"

# Fail early with a helpful message if the extension modules are not built,
# or were compiled against another scikit-learn. __check_build can be loaded
# before the version check because it only uses scikit-learn typedefs
from sklr import __check_build, _sklearn_version  # noqa: F401

# Public functions, defined in private modules of the subpackages
from sklr.utils._show_versions import show_versions

# Public subpackages, imported lazily on first attribute access
_submodules = ["utils"]

__all__ = ["show_versions"]
__all__.extend(_submodules)


def __dir__():
    return __all__


def __getattr__(name):
    if name in _submodules:
        return _importlib.import_module(f"sklr.{name}")
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"Module 'sklr' has no attribute {name!r}") from None
