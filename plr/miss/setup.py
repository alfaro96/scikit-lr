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
    """Configure the plr.miss module."""
    # Create the configuration file for the plr.miss module
    config = Configuration("miss", parent_package, top_path)

    # Initialize the libraries
    libraries = []

    # Fix POSIX Operating Systems including the macros in the library
    if name == "posix":
        libraries.append("m")

    # Add the tests subpackage
    config.add_subpackage("tests")

    # Return the configuration of the plr.miss module
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """Only called when this file is the "main" one."""
    # Setup the modules
    setup(**configuration().todict())
