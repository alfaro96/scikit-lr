"""Utilities useful during the build."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from distutils.version import LooseVersion


# =============================================================================
# Constants
# =============================================================================

# The following places must be sync
# with regard to the Cython version:
#   - .github/workflows/integration.yml
#   - docker/*/requirements/build.txt
#   - sklr/_build_utils/__init__.py
CYTHON_MIN_VERSION = "0.29.14"


# =============================================================================
# Methods
# =============================================================================

def _check_cython_version(min_version):
    """Check that the version of Cython installed in the system
    is greater than or equal the minimum required version."""
    # Re-raise with a more informative message when Cython is not installed
    try:
        import Cython
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install Cython with a version "
                                  ">= {0} in order to build scikit-lr from "
                                  "source.".format(min_version))

    if LooseVersion(Cython.__version__) < min_version:
        raise ValueError("The current version of Cython is {0} installed "
                         "in {1}.".format(Cython.__version__, Cython.__path__))


def cythonize_extensions(top_path, config):
    """Check that a recent version of Cython is
    available and cythonize the extension modules."""
    _check_cython_version(CYTHON_MIN_VERSION)

    # The cythonize method must be imported after checking
    # the Cython version so that a more informative message
    # can be provided to the user when Cython is not installed
    from Cython.Build import cythonize

    config.ext_modules = cythonize(config.ext_modules)
