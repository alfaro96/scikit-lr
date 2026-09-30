import subprocess
import sys

import pytest

from sklr._sklearn_version import check_sklearn_version


@pytest.mark.parametrize(
    "build_version, runtime_version",
    [
        ("1.9.1", "1.9.1"),
        ("1.9.0", "1.9.1"),
        ("1.9.0rc1", "1.9.1"),
        ("1.9.dev0", "1.9.0"),
        ("1.9.1", "1.9.1.post1"),
    ],
)
def test_check_sklearn_version_same_minor(build_version, runtime_version):
    check_sklearn_version(build_version, runtime_version)


@pytest.mark.parametrize(
    "build_version, runtime_version",
    [
        ("1.9.1", "1.10.0"),
        ("1.9.1", "1.8.2"),
        ("1.9.1", "2.9.1"),
        ("1.9.1", "1.10.dev0"),
        ("1.1.0", "1.10.0"),  # 1.1 is a string prefix of 1.10, not the same minor
    ],
)
def test_check_sklearn_version_different_minor(build_version, runtime_version):
    with pytest.raises(ImportError) as exc_info:
        check_sklearn_version(build_version, runtime_version)
    message = str(exc_info.value)
    assert f"built against scikit-learn {build_version}" in message
    assert f"scikit-learn {runtime_version} is installed" in message


def test_import_with_other_sklearn_version():
    # Run it in a fresh interpreter, where sklr has not been imported yet
    code = (
        "import sklearn\n"
        "sklearn.__version__ = '1.10.0'\n"
        "try:\n"
        "    import sklr\n"
        "except ImportError as error:\n"
        "    print(error)\n"
        "else:\n"
        "    raise SystemExit('sklr was imported')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "scikit-learn 1.10.0 is installed" in result.stdout
