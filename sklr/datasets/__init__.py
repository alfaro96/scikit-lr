"""The :mod:`sklr.datasets` module includes utilities to load datasets."""


# =============================================================================
# Imports
# =============================================================================

# Local application
from ._base import (load_authorship, load_blocks, load_bodyfat, load_breast,
                    load_calhousing, load_cold, load_cpu, load_diau, load_dtt,
                    load_ecoli, load_elevators, load_fried, load_glass,
                    load_heat, load_housing, load_iris, load_letter,
                    load_libras, load_pendigits, load_satimage, load_segment,
                    load_spo, load_stock, load_vehicle, load_vowel, load_wine,
                    load_wisconsin, load_yeast)


# =============================================================================
# Module public objects
# =============================================================================
__all__ = [
    "load_authorship", "load_blocks", "load_bodyfat", "load_breast",
    "load_calhousing", "load_cold", "load_cpu", "load_diau", "load_dtt",
    "load_ecoli", "load_elevators", "load_fried", "load_glass", "load_heat",
    "load_housing", "load_iris", "load_letter", "load_libras",
    "load_pendigits", "load_satimage", "load_segment", "load_spo",
    "load_stock", "load_vehicle", "load_vowel", "load_wine", "load_wisconsin",
    "load_yeast"
]
