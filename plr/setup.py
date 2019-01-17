# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy
from   numpy.distutils.core      import setup
from   numpy.distutils.misc_util import Configuration

# PLR
from plr._build_utils import maybe_cythonize_extensions

# Operating system
import os

# =============================================================================
# Global methods
# =============================================================================

def configuration(parent_package = "",
                  top_path       = None):
    """
        Configure the "plr" package.
    """
    # Create the configuration file
    config = Configuration("plr", parent_package, top_path)

    # Define the libraries
    libraries = []

    # Fix posix operating system 
    if os.name == "posix":
        libraries.append("m")

    # Add the submodules with build utilities
    config.add_subpackage("_build_utils")

    # Add the submodules with its own setup.py file
    config.add_subpackage("bucket")
    config.add_subpackage("metrics")
    config.add_subpackage("model_selection")
    config.add_subpackage("neighbors")
    config.add_subpackage("tree")
    config.add_subpackage("utils")

    # Add the data files
    config.add_data_files("_types.pxd")

    # Maybe cythonize the extension modules.
    # This is employed to avoid compiling
    # the Cython files in release mode
    maybe_cythonize_extensions(top_path, config)

    # Return the configuration 
    return config

if __name__ == '__main__':
    setup(**configuration(top_path = "").todict())
