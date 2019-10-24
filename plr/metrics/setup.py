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
    """Configure the plr.metrics module."""
    # Create the configuration file for plr.metrics module
    config = Configuration("metrics", parent_package, top_path)

    # Initialize the libraries
    libraries = []

    # Fix POSIX Operating Systems including the macros in the library
    if name == "posix":
        libraries.append("m")

    # Add the extension modules
    config.add_extension(
        "_label_ranking_fast",
        sources=["_label_ranking_fast.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"])

    config.add_extension(
        "_partial_label_ranking_fast",
        sources=["_partial_label_ranking_fast.pyx"],
        include_dirs=[get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"])

    # Add the data files
    config.add_data_files("_label_ranking_fast.pxd")
    config.add_data_files("_partial_label_ranking_fast.pxd")

    # Add the tests subpackage
    config.add_subpackage("tests")

    # Return the configuration of the plr.metrics module
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """Only called when this file is the "main" one."""
    # Setup the modules
    setup(**configuration().todict())
