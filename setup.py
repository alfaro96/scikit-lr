# =============================================================================
# Imports
# =============================================================================

# Standard
from setuptools import setup

# Local application
from sklr._build_utils import cythonize_extensions


# =============================================================================
# Constants
# =============================================================================

# The module name is used to locate the extension modules
MOD_NAME = "sklr"


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    setup(ext_modules=cythonize_extensions(MOD_NAME))
