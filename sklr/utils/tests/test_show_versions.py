import sklearn

from sklr import __version__, show_versions
from sklr._build_info import SKLEARN_BUILD_VERSION


def test_show_versions(capsys):
    show_versions()
    out = capsys.readouterr().out
    assert f"sklr: {__version__}" in out
    assert f"built with: scikit-learn {SKLEARN_BUILD_VERSION}" in out
    # Followed by the information of scikit-learn
    assert f"sklearn: {sklearn.__version__}" in out
    assert "Python dependencies:" in out
