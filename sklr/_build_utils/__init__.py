"""Utilities useful during the build."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from distutils.version import LooseVersion
import os
import sys

# Third party
import numpy as np


# =============================================================================
# Constants
# =============================================================================

# The following places must be sync
# with regard to the Cython version:
#   - .github/workflows/integration.yml
#   - docker/requirements/build.txt
#   - sklr/_build_utils/__init__.py
CYTHON_MIN_VERSION = "0.29.14"

# Extension modules that need to compile against NumPy should locate
# the corresponding include directory (source directory or core headers)
if np.show_config is None:
    d = os.path.join(os.path.dirname(np.__file__), "core", "include")
else:
    d = os.path.join(os.path.dirname(np.core.__file__), "include")

NUMPY_HEADERS_PATH = d

# Set the NumPy C API to a specific version instead of NPY_API_VERSION
# to avoid that the compilation is broken for released versions when
# NumPy introduces a new deprecation
NUMPY_NODEPR_API = dict(define_macros=[("NPY_NO_DEPRECATED_API",
                                        "NPY_1_9_API_VERSION")])


# =============================================================================
# Methods
# =============================================================================

def make_config(config):
    """Make the configuration of a module."""
    # Replace the point by a slash to list all the
    # directories and files in the current module
    path = config.name.replace(".", "/")
    (_, dirs, files) = next(os.walk(path))

    for module_dir in dirs:
        if module_dir == "data":
            config.add_data_dir(module_dir)
        else:
            config.add_subpackage(module_dir)

    for module_file in files:
        if module_file.endswith(".pxd"):
            config.add_data_files(module_file)
        elif module_file.endswith(".pyx"):
            add_extension(config, module_file)

    # Skip cythonization as it is not wanted to include the
    # generated C/C++ files in the release tarballs as they
    # are not necessarily, just for forward compatible with
    # future versions of Python, for instance
    if "sdist" not in sys.argv and config.name == "sklr":
        cythonize_extensions(config.top_path, config)

    return config


def add_extension(config, *sources):
    """Add an extension to the configuration."""
    (name, _) = os.path.splitext(sources[0])

    include_dirs = [NUMPY_HEADERS_PATH]

    # Specify the C++ version and ISO standard
    # for accessing to advanced features when
    # building the package for manylinux systems
    extra_link_args = ["-std=c++11"]
    extra_compile_args = ["-O3", "-std=c++11"]

    config.add_extension(name, sources,
                         language="c++",
                         include_dirs=include_dirs,
                         extra_link_args=extra_link_args,
                         extra_compile_args=extra_compile_args,
                         **NUMPY_NODEPR_API)


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
