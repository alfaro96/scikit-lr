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
    """Configure the plr.utils module."""
    # Create the configuration file for the plr.utils module
    config = Configuration("utils", parent_package, top_path)

    # Initialize the libraries
    libraries = []

    # Fix POSIX Operating Systems including the macros in the library
    if name == "posix":
        libraries.append("m")

    # Add the extension modules
    config.add_extension(
        "_argsort",
        sources=["_argsort.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"])

    config.add_extension(
        "_memory",
        sources=["_memory.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"])

    config.add_extension(
        "_ranking_fast",
        sources=["_ranking_fast.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3"])

    # Add the data files
    config.add_data_files("_argsort.pxd")
    config.add_data_files("_memory.pxd")
    config.add_data_files("_ranking_fast.pxd")

    # Add the tests subpackage
    config.add_subpackage("tests")

    # Return the configuration of the plr.utils module
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """Only called when this file is the "main" one."""
    # Setup the modules
    setup(**configuration().todict())
