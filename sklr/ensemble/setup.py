# =============================================================================
# Imports
# =============================================================================

# Standard
from os import name

# Third party
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


# =============================================================================
# Methods
# =============================================================================

def configuration(parent_package="", top_path=None):
    """Configure the sklr.ensemble module."""
    # Create the configuration file of the sklr.ensemble
    # module, where classes for ensemble methods will hold
    config = Configuration("ensemble", parent_package, top_path)

    # Initialize an empty list with the libraries that
    # must be included during the install process
    libraries = []

    # Fix POSIX Operating Systems (e.g., Unix) by
    # including the macros of th modules in the library
    if name == "posix":
        libraries.append("m")

    # Add the submodule with the
    # tests for the sklr.ensemble module
    config.add_subpackage("tests")

    # Return the configuration
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """Only called when this file is the "main" one."""
    setup(**configuration().todict())
