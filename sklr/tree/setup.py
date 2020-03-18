# =============================================================================
# Imports
# =============================================================================

# Third party
from numpy.distutils.misc_util import Configuration

# Local application
from sklr._build_utils import make_config


# =============================================================================
# Methods
# =============================================================================

def configuration(parent_package="", top_path=None):
    """Configure the module as a sub-package of the parent one."""
    return make_config(Configuration("tree", parent_package, top_path))
