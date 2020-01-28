# =============================================================================
# Imports
# =============================================================================

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

    # Add the submodule with tests
    # for the sklr.datasets module
    config.add_subpackage("tests")

    # Add the folder with the data
    config.add_data_dir("data")

    # Return the configuration file
    # of the sklr.datasets module
    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    setup(**configuration().todict())
