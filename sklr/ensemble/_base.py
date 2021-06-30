"""Base functions for ensemble-based estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np


# =============================================================================
# Functions
# =============================================================================

def _predict_ensemble(estimator, X, sample_weight=None):
    """Predict using an ensemble-based estimator."""
    X = estimator._validate_data(X, reset=False)
    aggregate = estimator._rank_algorithm.aggregate

    ensemble_Y = [estimator.predict(X) for estimator in estimator.estimators_]

    # TODO: ADD COMMENT HERE
    axes = (1, 0, 2)
    ensemble_Y = np.transpose(ensemble_Y, axes)
    print(ensemble_Y)

    Y = [aggregate(Y, sample_weight) for Y in ensemble_Y]

    return np.array(Y)
