import importlib
import os
import sys

import pytest

from sklr import __check_build
from sklr.__check_build import check_build, raise_build_error


def test_check_build():
    assert check_build() is None


def test_raise_build_error():
    error = ImportError("No module named 'sklr.__check_build._check_build'")
    with pytest.raises(ImportError, match="has not been built correctly") as exc_info:
        raise_build_error(error)
    assert exc_info.value.__cause__ is error
    assert "No module named 'sklr.__check_build._check_build'" in str(exc_info.value)
    # The message lists the contents of the directory to help debugging
    assert os.path.dirname(__check_build.__file__) in str(exc_info.value)
    assert "__init__.py" in str(exc_info.value)


def test_import_without_extension_module(monkeypatch):
    # A None entry in sys.modules makes importing that module raise ImportError
    monkeypatch.setitem(sys.modules, "sklr.__check_build._check_build", None)
    with pytest.raises(ImportError, match="has not been built correctly"):
        importlib.reload(__check_build)
