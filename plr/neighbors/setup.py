# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy
from   numpy.distutils.core      import setup
from   numpy.distutils.misc_util import Configuration

# Operating System
import os

# =============================================================================
# Global methods
# =============================================================================

def configuration(parent_package = "",
                  top_path       = None):
    """
        Configure "neighbors" package.
    """
    # Initialize the configuration for the "neighbors" package
    config = Configuration("neighbors", parent_package, top_path)
    libraries = []

    # Fix posix operating system 
    if os.name == "posix":
        libraries.append("m")

    # Add the tests subpackage
    config.add_subpackage("tests")

    # Return the configuration for the "neighbors" package
    return config

# Only called if this file is the main one
if __name__ == "__main__":
    # Setup the files
    setup(**configuration().todict())
