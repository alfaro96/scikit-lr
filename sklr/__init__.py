"""Label ranking and partial label ranking with scikit-learn compatible estimators."""

import importlib as _importlib

# This line is also parsed by sklr/_build_utils/version.py to set the
# version of the distribution, so keep it as a single string literal.
__version__ = "0.3.dev0"

# Public subpackages, imported lazily on first attribute access
_submodules: list[str] = []

__all__ = list(_submodules)


def __dir__():
    return __all__


def __getattr__(name):
    if name in _submodules:
        return _importlib.import_module(f"sklr.{name}")
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"Module 'sklr' has no attribute {name!r}") from None
