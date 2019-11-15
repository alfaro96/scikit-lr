"""The :mod:`sklr.utils` module includes various utilities."""


# =============================================================================
# Imports
# =============================================================================

# Local application
from .bunch import Bunch
from .ranking import (
    unique_rankings, check_label_ranking_targets,
    check_partial_label_ranking_targets, type_of_targets,
    is_ranking_without_ties, is_ranking_with_ties, rank_data)
from .validation import (
    check_array, check_is_fitted, check_consistent_length,
    check_random_state, check_X_Y, has_fit_parameter)


# =============================================================================
# Public objects
# =============================================================================

# Set the classes that are accessible
# from the module sklr.utils
__all__ = ["Bunch"]

# Set the methods that are accessible
# from the module sklr.utils
__all__ += [
    "check_array", "check_consistent_length",
    "check_is_fitted", "check_label_ranking_targets",
    "check_partial_label_ranking_targets", "check_random_state",
    "check_X_Y", "has_fit_parameter", "is_ranking_without_ties",
    "is_ranking_with_ties", "rank_data", "type_of_targets", "unique_rankings"
]
