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
    """Configure the sklr.utils module."""
    # Create the configuration file of the sklr.utils module,
    # where classes and methods for various utilities will hold
    config = Configuration("utils", parent_package, top_path)

    # Initialize an empty list with the libraries that
    # must be included during the install process
    libraries = []

    # Fix POSIX Operating Systems (e.g., Unix) by
    # including the macros of th modules in the library
    if name == "posix":
        libraries.append("m")

    # Add the extension modules, setting the programming language
    # and some extra arguments needed during the compilation

    # Argsort
    config.add_extension(
        "_argsort",
        sources=["_argsort.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"])

    # Memory
    config.add_extension(
        "_memory",
        sources=["_memory.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"])

    # Ranking
    config.add_extension(
        "_ranking_fast",
        sources=["_ranking_fast.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
        extra_link_args=["-std=c++11"])

    # Add the data files
    config.add_data_files("_argsort.pxd")
    config.add_data_files("_memory.pxd")
    config.add_data_files("_ranking_fast.pxd")

    # Add the submodule with the
    # tests for the sklr.utils module
    config.add_subpackage("tests")

    # Return the configuration
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """Only called when this file is the "main" one."""
    setup(**configuration().todict())
