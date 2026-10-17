"""Utilities for the tests of scikit-lr."""

import importlib
import inspect
import pkgutil

from sklearn.base import BaseEstimator

import sklr
from sklr.base import is_label_ranker, is_partial_label_ranker


def public_modules(module=sklr):
    """Yield `module` and its public submodules, recursively."""
    yield module
    # Only a package has a __path__ in which to look for submodules
    path = getattr(module, "__path__", [])
    for module_info in pkgutil.iter_modules(path, f"{module.__name__}."):
        name = module_info.name.rpartition(".")[2]
        if not name.startswith("_") and name not in {"conftest", "tests"}:
            yield from public_modules(importlib.import_module(module_info.name))


def all_rankers():
    """Return the public label rankers and partial label rankers, sorted by name."""
    rankers = {
        obj
        for module in public_modules()
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if not name.startswith("_")
        and obj.__module__.partition(".")[0] == "sklr"
        and issubclass(obj, BaseEstimator)
        and (is_label_ranker(obj()) or is_partial_label_ranker(obj()))
    }
    return sorted(rankers, key=lambda ranker: ranker.__name__)
