"""
The :mod:`sklr.datasets` module includes utilities to
load popular reference datasets.
"""


# =============================================================================
# Imports
# =============================================================================

# Local application
from ._base import (
    load_authorship, load_bodyfat, load_blocks, load_breast,
    load_calhousing, load_cold, load_cpu, load_diau, load_dtt,
    load_ecoli, load_elevators, load_fried, load_glass, load_heat,
    load_housing, load_iris, load_letter, load_libras, load_pendigits,
    load_satimage, load_segment, load_shuttle, load_spo, load_stock,
    load_vehicle, load_vowel, load_wine, load_wisconsin, load_yeast)


# =============================================================================
# Public objects
# =============================================================================

# Set the method that are accessible
# from the module sklr.datasets
__all__ = [
    "load_authorship", "load_bodyfat", "load_blocks", "load_breast",
    "load_calhousing", "load_cold", "load_cpu", "load_diau",
    "load_dtt", "load_ecoli", "load_elevators", "load_fried",
    "load_glass", "load_heat", "load_housing", "load_iris",
    "load_letter", "load_libras", "load_pendigits", "load_satimage",
    "load_segment", "load_shuttle", "load_spo", "load_stock",
    "load_vehicle", "load_vowel", "load_wine", "load_wisconsin", "load_yeast"
]
