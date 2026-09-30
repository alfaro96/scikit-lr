import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest

import sklr


def test_import_star():
    # "import *" is only allowed at module level, so run it in a fresh interpreter
    subprocess.run([sys.executable, "-c", "from sklr import *"], check=True)


def test_version_matches_distribution_metadata():
    # The distribution takes its version from sklr/__init__.py at build time
    assert sklr.__version__ == importlib.metadata.version("scikit-lr")


def test_version_script():
    script = Path(sklr.__file__).parent / "_build_utils" / "version.py"
    result = subprocess.run(
        [sys.executable, script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == sklr.__version__


def test_dir_lists_public_api():
    assert sorted(dir(sklr)) == sorted(sklr.__all__)


def test_getattr_unknown_name():
    with pytest.raises(AttributeError, match="Module 'sklr' has no attribute 'foo'"):
        sklr.foo  # noqa: B018
