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
    """Configure the sklr.datasets module."""
    # Create the configuration file of the sklr.datasets module,
    # where methods to load popular reference datasets will hold
    config = Configuration("datasets", parent_package, top_path)

    # Initialize an empty list with the libraries that
    # must be included during the install process
    libraries = []

    # Fix POSIX Operating Systems (e.g., Unix) by
    # including the macros of th modules in the library
    if name == "posix":
        libraries.append("m")

    # Add the submodule wit the test
    # for the sklr.datasets module
    config.add_subpackage("tests")

    # Add the folder with the data
    config.add_data_dir("data")

    # Return the configuration
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    """Only called when this file is the "main" one."""
    setup(**configuration().todict())
