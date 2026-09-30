import os

import pytest

from sklr import __check_build
from sklr.__check_build import raise_build_error


def test_raise_build_error():
    error = ImportError("No module named 'sklr.__check_build._check_build'")
    with pytest.raises(ImportError, match="has not been built correctly") as exc_info:
        raise_build_error(error)
    assert exc_info.value.__cause__ is error
    assert "No module named 'sklr.__check_build._check_build'" in str(exc_info.value)
    # The message lists the contents of the directory to help debugging
    assert os.path.dirname(__check_build.__file__) in str(exc_info.value)
    assert "__init__.py" in str(exc_info.value)
