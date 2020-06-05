"""
Machine Learning module for Label Ranking problems in Python
============================================================

Scikit-lr is a Python module integrating Machine Learning algorithms
for Label Ranking problems in the tightly-knit world of Scientific
Python packages.

It aims is to provide simple and efficient solutions to Label Ranking
problems that are accessible to everybody and reusable in all contexts.
"""


# =============================================================================
# Constants
# =============================================================================

# PEP0440 compatible formatted version
#
# Generic release markers:
#   * X.Y
#   * X.Y.Z     # Bug-fix release
#
# Admissible pre-release markers:
#   * X.YaN     # Alpha release
#   * X.YbN     # Beta release
#   * X.YrcN    # Release candidate
#   * X.Y       # Final release
#
# Developmental release markers:
#   * X.Y.dev   # Canonical of X.Y.dev0
#   * X.Y.devN

# Scikit-lr package version
__version__ = "0.3.dev0"


# =============================================================================
# Module public objects
# =============================================================================
__all__ = [
    "consensus",
    "datasets",
    "dummy",
    "ensemble",
    "exceptions",
    "metrics",
    "miss",
    "neighbors",
    "tree",
    "utils"
]
