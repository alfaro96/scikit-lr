"""
    The module "plr.metrics" includes several metrics
    for probability distributions and rankings.
"""

# =============================================================================
# Imports
# =============================================================================

# Probability distributions
from .probability import bhattacharyya_distance, bhattacharyya_score

# Rankings
from .ranking import kendall_distance, tau_x_score

# Define the objects to be exported
__all__ = ["bhattacharyya_distance",
           "bhattacharyya_score",
           "kendall_distance",
           "tau_x_score"]
