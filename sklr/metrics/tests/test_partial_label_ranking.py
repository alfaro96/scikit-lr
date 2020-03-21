"""Testing of Partial Label Ranking metrics."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from scipy.stats import rankdata
import numpy as np
import pytest

# Local application
from sklr.metrics import tau_x_score
from sklr.utils import check_random_state


# =============================================================================
# Methods
# =============================================================================

def _tau_x_matrix(Y, sample, f_class, s_class):
    """Matrix for the Kendall tau extension."""
    return (0 if f_class == s_class else
            -1 if Y[sample, f_class] > Y[sample, s_class] else 1)


def _tau_x_score(Y_true, Y_pred, sample_weight=None):
    """Alternative implementation of Kendall tau extension.

    This implementation follows the original article's definition (see
    References). This should give identical results as ``tau_x_score``.

    References
    ----------
    .. [1] `E. J. Emond and D. W. Mason, "A new rank correlation coefficient
            with application to the consensus ranking problem", Journal of
            Multi-Criteria Decision Analysis, vol. 11, pp. 17-28, 2002.`_
    """
    (n_samples, n_classes) = Y_true.shape
    scores = np.zeros(n_samples)

    for sample in range(n_samples):
        for f_class in range(n_classes):
            for s_class in range(n_classes):
                a = _tau_x_matrix(Y_true, sample, f_class, s_class)
                b = _tau_x_matrix(Y_pred, sample, f_class, s_class)
                scores[sample] += a * b

        scores[sample] /= n_classes * (n_classes-1)

    return np.average(a=scores, weights=sample_weight)


def _make_partial_label_ranking(n_samples, n_classes, random_state):
    """Helper method to make a Partial Label Ranking problem."""
    rankings = np.zeros((n_samples, n_classes), dtype=np.int64)

    for sample in range(n_samples):
        rankings[sample] = random_state.choice(n_samples, n_classes, True)
        rankings[sample] = rankdata(rankings[sample], method="dense")

    return rankings


def _make_sample_weights(n_repetitions, n_samples, random_state):
    """Helper method to make random sample weights."""
    sample_weights = np.zeros((n_repetitions, n_samples), dtype=np.float64)

    for repetition in range(n_repetitions):
        sample_weights[repetition] = random_state.rand(n_samples) + 1

    return sample_weights


# =============================================================================
# Initialization
# =============================================================================

# Initialize a random number generator to
# ensure that the tests are reproducible
seed = 198075
random_state = check_random_state(seed)

n_repetitions = 2
n_samples = 20
n_classes = 5

Y_true = _make_partial_label_ranking(n_samples, n_classes, random_state)
Y_pred = _make_partial_label_ranking(n_samples, n_classes, random_state)
sample_weights = _make_sample_weights(n_repetitions, n_samples, random_state)


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.parametrize("sample_weight", sample_weights)
def test_tau_x_score(sample_weight):
    """Test the tau_x_score method."""
    np.testing.assert_almost_equal(
        tau_x_score(Y_true, Y_pred, sample_weight),
        _tau_x_score(Y_true, Y_pred, sample_weight))
