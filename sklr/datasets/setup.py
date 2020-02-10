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
    """Configure the module."""
    config = Configuration("datasets", parent_package, top_path)

    # Need to include the data files to ensure its
    # availability when creating the distributions
    config.add_data_dir("data")
    config.add_subpackage("tests")

    return config


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    setup(**configuration(top_path="").todict())
