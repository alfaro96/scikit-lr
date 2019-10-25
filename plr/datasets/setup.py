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
    """Configure the plr.datasets module."""
    # Create the configuration file for the plr.datasets module
    config = Configuration("datasets", parent_package, top_path)

    # Initialize the libraries
    libraries = []

    # Fix POSIX Operating Systems including the macros in the library
    if name == "posix":
        libraries.append("m")

    # Add the tests subpackage
    config.add_subpackage("tests")

    # Add the data folder
    config.add_data_dir("data")

    # Return the configuration of the plr.datasets module
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """Only called when this file is the "main" one."""
    # Setup the modules
    setup(**configuration().todict())
