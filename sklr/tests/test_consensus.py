"""Testing for consensus classes."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.consensus import RankAggregationAlgorithm


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.parametrize("algorithm", ["foo", "borda_count", "bpa_lia_mp2"])
@pytest.mark.get_algorithm
def test_get_algorithm(algorithm):
    """Test the get_algorithm method for RankAggregationAlgorithm."""
    # Assert that an exception is raised for
    # the "foo" algorithm, that is not available
    if algorithm == "foo":
        with pytest.raises(ValueError):
            RankAggregationAlgorithm.get_algorithm(algorithm)

    # Assert that the proper RankAggregationAlgorithm
    # object is returned for existing algorithms
    else:
        assert issubclass(
            type(RankAggregationAlgorithm.get_algorithm(algorithm)),
            RankAggregationAlgorithm)


@pytest.mark.aggregate_borda_count
def test_aggregate_borda_count():
    """Test the aggregate method for BordaCountAlgorithm."""
    # Initialize the BordaCountAlgorithm object
    borda_count = RankAggregationAlgorithm.get_algorithm("borda_count")

    # Initialize the rankings
    # (complete and with randomly missed classes)
    Y = np.array([[1, 2, 3], [2, 3, 1], [2, 3, 1]])
    Y_random = np.array([[1, 2, np.nan], [2, np.nan, 1], [2, 3, 1]])

    # Initialize the true and predicted consensus rankings
    # (for the complete rankings and for the
    # rankings with randomly missed classes)
    consensus_true = np.array([1, 3, 2])
    consensus_pred = borda_count.aggregate(Y)
    consensus_random_true = np.array([2, 3, 1])
    consensus_random_pred = borda_count.aggregate(Y_random)

    # Assert that the consensus rankings are correct
    np.testing.assert_array_equal(consensus_pred, consensus_true)
    np.testing.assert_array_equal(consensus_random_pred, consensus_random_true)

    # Also, compute the consensus ranking from the rankings
    # with randomly missed classes using the MLE process and
    # return the completed rankings to ensure that they are correct
    consensus_random_true = np.array([2, 3, 1])
    Y_completed_true = np.array([[2, 3, 1], [2, 3, 1], [2, 3, 1]])
    (consensus_random_pred, Y_completed_pred) = borda_count.aggregate(
        Y_random, apply_mle=True, return_Yt=True)

    # Assert that the consensus ranking and
    # the completed rankings are correct
    np.testing.assert_array_equal(consensus_random_pred, consensus_random_true)
    np.testing.assert_array_equal(Y_completed_pred, Y_completed_true)


@pytest.mark.aggregate_bpa_lia_mp2
def test_aggregate_bpa_lia_mp2():
    """Test the aggregate method for BPALIAMP2Algorithm."""
    # Initialize the BPALIAMP2Algorithm object
    bpa_lia_mp2 = RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2")

    # Initialize the rankings
    # (complete and with randomly missed classes)
    Y = np.array([[1, 2, 2], [2, 2, 1], [2, 1, 1]])
    Y_random = np.array([[1, 2, np.nan], [2, np.nan, 1], [2, 1, 1]])

    # Initialize the true and predicted consensus rankings
    # (for the complete rankings and for the
    # rankings with randomly missed classes)
    consensus_true = np.array([1, 1, 1])
    consensus_pred = bpa_lia_mp2.aggregate(Y)
    consensus_random_true = np.array([1, 1, 1])
    consensus_random_pred = bpa_lia_mp2.aggregate(Y_random)

    # Assert that the consensus rankings are correct
    np.testing.assert_array_equal(consensus_pred, consensus_true)
    np.testing.assert_array_equal(consensus_random_pred, consensus_random_true)

    # Assert that a proper error is raised when trying
    # to compute the consensus ranking using the MLE process
    with pytest.raises(NotImplementedError):
        bpa_lia_mp2.aggregate(Y_random, apply_mle=True, return_Yt=True)


@pytest.mark.sample_weight_invalid
@pytest.mark.parametrize("algorithm", ["borda_count", "bpa_lia_mp2"])
def test_sample_weight_invalid(algorithm):
    """Test that invalid sample weighting raises errors."""
    # Initialize the RankAggregationAlgorithm object
    algorithm = RankAggregationAlgorithm.get_algorithm(algorithm)

    # Initialize the rankings that will be employed to assert that
    # the different sample weights that are provided are invalid
    Y = np.array([[1, 2, 2], [2, 2, 1], [2, 1, 1]])
    n_samples = Y.shape[0]

    # Initialize the different sample weights to be tested
    sample_weight_zero = np.array(0)
    sample_weight_2d = np.ones((n_samples, 1))
    sample_weight_minus = np.ones(n_samples - 1)
    sample_weight_plus = np.ones(n_samples + 1)

    # Assert that an error is raised when the
    # sample weights are not a 1-D array
    with pytest.raises(ValueError):
        algorithm.aggregate(Y, sample_weight_zero)
    with pytest.raises(ValueError):
        algorithm.aggregate(Y, sample_weight_2d)

    # Assert that an error is raised when the
    # sample weights have wrong number of samples
    with pytest.raises(ValueError):
        algorithm.aggregate(Y, sample_weight_minus)
    with pytest.raises(ValueError):
        algorithm.aggregate(Y, sample_weight_plus)
