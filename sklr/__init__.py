"""
Machine Learning package for Label Ranking problems in Python
=============================================================

Scikit-lr is a Python package integrating Machine Learning algorithms
for Label Ranking problems in the tightly-knit world of Scientific Python
packages.

It aims to provide simple and efficient solutions to Label Ranking
problems that are accessible to everybody and reusable in all contexts.

See: https://scikit-lr.readthedocs.io for complete documentation.
"""


# =============================================================================
# Constants
# =============================================================================

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440
#
# Generic release markers:
#   * X.Y
#   * X.Y.Z   # Bug-fix release
#
# Admissible pre-release markers:
#   * X.YaN   # Alpha release
#   * X.YbN   # Beta release
#   * X.YrcN  # Release candidate
#   * X.Y     # Final release
#
# Dev branch marker is: "X.Y.dev" or "X.Y.devN", where N is an integer.
# "X.Y.dev0" is the canonical version of "X.Y.dev".

# Scikit-lr package version
__version__ = "0.3.dev0"


# =============================================================================
# Module public objects
# =============================================================================
__all__ = [
    "consensus",
    "dummy",
    "metrics",
    "neighbors",
    "tree",
    "utils"
]
