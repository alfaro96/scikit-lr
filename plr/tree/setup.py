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
    """Configure the plr.tree module."""
    # Create the configuration file for the plr.tree module
    config = Configuration("tree", parent_package, top_path)

    # Initialize the libraries
    libraries = []

    # Fix POSIX Operating Systems including the macros in the library
    if name == "posix":
        libraries.append("m")

    # Add the extension modules
    config.add_extension(
        "_criterion",
        sources=["_criterion.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
        extra_link_args=["-std=c++11"])

    config.add_extension(
        "_splitter",
        sources=["_splitter.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
        extra_link_args=["-std=c++11"])

    config.add_extension(
        "_tree",
        sources=["_tree.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
        extra_link_args=["-std=c++11"])

    # Add the data files
    config.add_data_files("_criterion.pxd")
    config.add_data_files("_splitter.pxd")
    config.add_data_files("_tree.pxd")

    # Add the tests subpackage
    config.add_subpackage("tests")

    # Return the configuration of the plr.tree module
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """Only called when this file is the "main" one."""
    # Setup the modules
    setup(**configuration().todict())
