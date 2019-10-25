# =============================================================================
# Imports
# =============================================================================

# Standard
from os import name

# Third party
from numpy import get_include
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

# Local application
from plr._build_utils import maybe_cythonize_extensions


# =============================================================================
# Methods
# =============================================================================

def configuration(parent_package="", top_path=None):
    """Configure the plr module."""
    # Create the configuration for the plr package
    config = Configuration("plr", parent_package, top_path)

    # Initialize the libraries
    libraries = []

    # Fix POSIX Operating Systems including the macros in the library
    if name == "posix":
        libraries.append("m")

    # Add the extension modules
    config.add_extension(
        "consensus",
        sources=["consensus.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
        extra_link_args=["-std=c++11"])

    # Add the data files
    config.add_data_files("consensus.pxd")
    config.add_data_files("_types.pxd")

    # Add the submodules with build utilities
    config.add_subpackage("_build_utils")

    # Add the submodules with its own setup.py file
    config.add_subpackage("datasets")
    config.add_subpackage("ensemble")
    config.add_subpackage("metrics")
    config.add_subpackage("miss")
    config.add_subpackage("neighbors")
    config.add_subpackage("tree")
    config.add_subpackage("utils")

    # Add the tests subpackage
    config.add_subpackage("tests")

    # Maybe "cythonize" the extension modules.
    # This is employed to avoid compiling
    # the Cython files in release mode
    maybe_cythonize_extensions(top_path, config)

    # Return the configuration of the plr package
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """Only called when this file is the "main" one."""
    # Setup the modules
    setup(**configuration().todict())
