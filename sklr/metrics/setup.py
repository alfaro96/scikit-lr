# =============================================================================
# Imports
# =============================================================================

# Standard
from os import name

# Third party
from numpy import get_include
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


# =============================================================================
# Methods
# =============================================================================

def configuration(parent_package="", top_path=None):
    """Configure the sklr.metrics module."""
    # Create the configuration file for sklr.metrics module,
    # where classes and methods for performnace will hold
    config = Configuration("metrics", parent_package, top_path)

    # Initialize an empty list with the libraries that
    # must be included during the install process
    libraries = []

    # Fix POSIX Operating Systems (e.g., Unix) by
    # including the macros of th modules in the library
    if name == "posix":
        libraries.append("m")

    # Add the extension modules, setting the programming language
    # and some extra arguments needed during the compilation

    # Label Ranking fast
    config.add_extension(
        "_label_ranking_fast",
        sources=["_label_ranking_fast.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"])

    # Partial Label Ranking fast
    config.add_extension(
        "_partial_label_ranking_fast",
        sources=["_partial_label_ranking_fast.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"])

    # Add the data files
    config.add_data_files("_label_ranking_fast.pxd")
    config.add_data_files("_partial_label_ranking_fast.pxd")

    # Add the submodule with tests
    # for the sklr.metrics module
    config.add_subpackage("tests")

    # Return the configuration
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """Only called when this file is the "main" one."""
    setup(**configuration().todict())
