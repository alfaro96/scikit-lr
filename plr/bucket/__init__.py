"""
    The module "plr.bucket" module includes objects
    related with the Optimal Bucket Order Problem
"""

# =============================================================================
# Imports
# =============================================================================

# Matrices
from .matrix import PairOrderMatrix, UtopianMatrix, AntiUtopianMatrix

# Optimal Bucket Order Problem
from .obop import OptimalBucketOrderProblem

# Bucket order
from .order import BucketOrder

# =============================================================================
# Public objects
# =============================================================================

# Classes
__all__ = ["BucketOrder",
           "PairOrderMatrix",
           "UtopianMatrix",
           "AntiUtopianMatrix",
           "OptimalBucketOrderProblem"]
