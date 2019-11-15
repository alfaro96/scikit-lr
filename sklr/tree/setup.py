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
    """Configure the sklr.tree module."""
    # Create the configuration file of the sklr.utils module,
    # where classes for decision tree estimators will hold
    config = Configuration("tree", parent_package, top_path)

    # Initialize an empty list with the libraries that
    # must be included during the install process
    libraries = []

    # Fix POSIX Operating Systems (e.g., Unix) by
    # including the macros of th modules in the library
    if name == "posix":
        libraries.append("m")

    # Add the extension modules, setting the programming language
    # and some extra arguments needed during the compilation

    # Criterion
    config.add_extension(
        "_criterion",
        sources=["_criterion.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
        extra_link_args=["-std=c++11"])

    # Splitter
    config.add_extension(
        "_splitter",
        sources=["_splitter.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
        extra_link_args=["-std=c++11"])

    # Tree
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

    # Add the submodule with the
    # tests for the sklr.tree module
    config.add_subpackage("tests")

    # Return the configuration
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """Only called when this file is the "main" one."""
    setup(**configuration().todict())
