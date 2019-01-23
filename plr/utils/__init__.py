"""
    The module "plr.utils" includes utils
    for different kind of operations.
"""

# =============================================================================
# Imports
# =============================================================================

# Datasets
from .datasets import ClusterProbability, TopK, MissRandom

# Validation
from .validation import check_n_features, check_is_fitted, check_is_type, check_prob_dists, check_random_state, check_true_pred_sample_weight, check_X, check_X_Y, check_Y, check_Y_prob_dists, check_X_Y_sample_weight

# =============================================================================
# Public objects
# =============================================================================

# Classes
__all__ = ["ClusterProbability",
           "MissRandom",
           "TopK"]

# Methods
__all__ += ["check_arrays",
            "check_is_fitted",
            "check_is_type",
            "check_n_features",
            "check_prob_dists",
            "check_random_state",
            "check_true_pred_sample_weight",
            "check_X",
            "check_X_Y",
            "check_X_Y_sample_weight",
            "check_Y",
            "check_Y_prob_dists",
            "check_Y_sample_weight"]
