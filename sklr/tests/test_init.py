import importlib.metadata
import subprocess
import sys

import pytest

import sklr


def test_import_star():
    # "import *" is only allowed at module level, so run it in a fresh interpreter
    subprocess.run([sys.executable, "-c", "from sklr import *"], check=True)


def test_version_matches_distribution_metadata():
    # The distribution takes its version from sklr/__init__.py at build time,
    # read by sklr/_build_utils/version.py, so this also checks that script
    assert sklr.__version__ == importlib.metadata.version("scikit-lr")


def test_dir_lists_public_api():
    assert sorted(dir(sklr)) == sorted(sklr.__all__)


def test_getattr_imports_submodule(monkeypatch):
    # A subpackage of the tests stands in for a public one, which may not
    # exist. Once imported, it is an attribute of sklr, so __getattr__ is
    # called directly
    monkeypatch.setattr(sklr, "_submodules", ["tests"])
    assert sklr.__getattr__("tests") is sys.modules["sklr.tests"]


def test_getattr_unknown_name():
    with pytest.raises(AttributeError, match="Module 'sklr' has no attribute 'foo'"):
        sklr.foo  # noqa: B018
