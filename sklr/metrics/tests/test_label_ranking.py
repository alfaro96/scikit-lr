"""Testing of Label Ranking metrics."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.metrics import kendall_distance, tau_score
from sklr.utils import check_random_state


# =============================================================================
# Methods
# =============================================================================

def _kendall_distance(Y_true, Y_pred, normalize=True, sample_weight=None):
    """Alternative implementation of the Kendall distance.

    This implementation follows the Wikipedia article's definition (see
    References). This should give identical results as ``kendall_distance``.

    References
    ----------
    .. [1] `Wikipedia entry for the Kendall tau distance.
            <https://en.wikipedia.org/wiki/Kendall_tau_distance>`_
    """
    (n_samples, n_classes) = Y_true.shape
    dists = np.zeros(n_samples)

    for sample in range(n_samples):
        for f_class in range(n_classes - 1):
            for s_class in range(f_class + 1, n_classes):
                a = Y_true[sample, f_class] - Y_true[sample, s_class]
                b = Y_pred[sample, f_class] - Y_pred[sample, s_class]

                if a * b < 0:
                    dists[sample] += 1

        if normalize:
            dists[sample] /= n_classes * (n_classes-1) / 2

    return np.average(a=dists, weights=sample_weight)


def _tau_score(Y_true, Y_pred, sample_weight=None):
    """Alternative implementation of the Kendall tau.

    This implementation follows the Wikipedia article's definition (see
    References). This should give identical results as ``tau_score``.

    References
    ----------
    .. [1] `Wikipedia entry for the Kendall rank correlation coefficient.
            <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_
    """
    (n_samples, n_classes) = Y_true.shape
    scores = np.zeros(n_samples)

    for sample in range(n_samples):
        for f_class in range(n_classes - 1):
            for s_class in range(f_class + 1, n_classes):
                a = Y_true[sample, f_class] - Y_true[sample, s_class]
                b = Y_pred[sample, f_class] - Y_pred[sample, s_class]
                scores[sample] += np.sign(a * b)

        scores[sample] *= 2 / (n_classes * (n_classes-1))

    return np.average(a=scores, weights=sample_weight)


def _make_label_ranking(n_samples, n_classes, random_state):
    """Helper method to make a Label Ranking problem."""
    rankings = np.zeros((n_samples, n_classes), dtype=np.int64)

    for sample in range(n_samples):
        rankings[sample] = random_state.permutation(n_classes) + 1

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

# Values that the "normalize" parameter can take to
# parametrize the testing method of kendall_distance
NORMALIZE = [True, False]

# Initialize a random number generator to
# ensure that the tests are reproducible
seed = 198075
random_state = check_random_state(seed)

n_repetitions = 2
n_samples = 20
n_classes = 5

Y_true = _make_label_ranking(n_samples, n_classes, random_state)
Y_pred = _make_label_ranking(n_samples, n_classes, random_state)
sample_weights = _make_sample_weights(n_repetitions, n_samples, random_state)


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.parametrize("normalize", NORMALIZE)
@pytest.mark.parametrize("sample_weight", sample_weights)
def test_kendall_distance(normalize, sample_weight):
    """Test the kendall_distance method."""
    np.testing.assert_almost_equal(
        kendall_distance(Y_true, Y_pred, normalize, sample_weight),
        _kendall_distance(Y_true, Y_pred, normalize, sample_weight))


@pytest.mark.parametrize("sample_weight", sample_weights)
def test_tau_score(sample_weight):
    """Test the tau_score method."""
    np.testing.assert_almost_equal(
        tau_score(Y_true, Y_pred, sample_weight),
        _tau_score(Y_true, Y_pred, sample_weight))
