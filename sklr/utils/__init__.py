"""The :mod:`sklr.utils` module includes various utilities."""


# =============================================================================
# Imports
# =============================================================================

# Local application
from .ranking import (check_label_ranking_targets,
                      check_partial_label_ranking_targets, type_of_target)
from .validation import (check_array, check_is_fitted, check_consistent_length,
                         check_random_state, check_sample_weight, check_X_Y,
                         has_fit_parameter)


# =============================================================================
# Module public objects
# =============================================================================
__all__ = [
    "check_array", "check_consistent_length", "check_is_fitted",
    "check_label_ranking_targets", "check_partial_label_ranking_targets",
    "check_random_state", "check_sample_weight", "check_X_Y",
    "has_fit_parameter", "type_of_target"
]
