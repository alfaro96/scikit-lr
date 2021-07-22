"""."""


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

MOD_NAME = "sklr"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Cythonize to generate the extension modules
    ext_modules = cythonize_extensions(MOD_NAME)

    setup(ext_modules=ext_modules)
