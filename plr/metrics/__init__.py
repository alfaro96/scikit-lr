"""
    The module "plr.metrics" includes several metrics
    for different kind of data.
"""

# =============================================================================
# Imports
# =============================================================================

# Probability distributions
from .probability import bhattacharyya_distance, bhattacharyya_score

# Rankings
from .ranking import kendall_distance, tau_x_score

# Spatial
from .spatial import minkowski

# Define the objects to be exported
__all__ = ["bhattacharyya_distance",
           "bhattacharyya_score",
           "kendall_distance",
           "tau_x_score",
           "minkowski"]
