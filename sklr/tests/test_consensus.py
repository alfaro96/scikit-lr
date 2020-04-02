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
    """Test the "get_algorithm" method for "RankAggregationAlgorithm"."""
    # Assert that the proper Rank Aggregation
    # algorithm is returned for existing algorithms
    if algorithm != "foo":
        assert issubclass(
            type(RankAggregationAlgorithm.get_algorithm(algorithm)),
            RankAggregationAlgorithm)
    # Assert that the proper error is raised
    # for the not available "foo" algorithm
    else:
        with pytest.raises(ValueError):
            RankAggregationAlgorithm.get_algorithm(algorithm)


@pytest.mark.aggregate_borda_count
def test_aggregate_borda_count():
    """Test the "aggregate" method for "BordaCountAlgorithm"."""
    # Initialize the BordaCountAlgorithm object
    borda_count = RankAggregationAlgorithm.get_algorithm("borda_count")

    # Initialize the rankings (complete and with missed classes)
    Y = np.array([[1, 2, 3], [2, 3, 1], [2, 3, 1]])
    Y_random = np.array([[1, 2, np.nan], [2, np.nan, 1], [2, 3, 1]])
    Y_top = np.array([[1, 2, np.inf], [2, np.inf, 1], [2, np.inf, 1]])

    # Initialize the true consensus rankings, that is, those
    # that must be obtained from the rankings previously defined
    consensus_true = np.array([1, 3, 2])
    consensus_random_true = np.array([2, 3, 1])
    consensus_top_true = np.array([1, 3, 2])

    # Initialize the predicted consensus ranking, that is,
    # those obtained using the Borda count algorithm
    consensus_pred = borda_count.aggregate(Y)
    consensus_random_pred = borda_count.aggregate(Y_random)
    consensus_top_pred = borda_count.aggregate(Y_top)

    # Assert that the consensus rankings are the same
    np.testing.assert_array_equal(consensus_pred, consensus_true)
    np.testing.assert_array_equal(consensus_random_pred, consensus_random_true)
    np.testing.assert_array_equal(consensus_top_pred, consensus_top_true)


@pytest.mark.aggregate_bpa_lia_mp2
def test_aggregate_bpa_lia_mp2():
    """Test the "aggregate" method for "BPALIAMP2Algorithm"."""
    # Initialize the BPALIAMP2Algorithm object
    bpa_lia_mp2 = RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2")

    # Initialize the rankings (complete and with missed classes)
    Y = np.array([[1, 2, 2], [2, 2, 1], [2, 1, 1]])
    Y_random = np.array([[1, 2, np.nan], [2, np.nan, 1], [2, 1, 1]])
    Y_top = np.array([[1, 2, np.inf], [2, np.inf, 1], [np.inf, 1, 1]])

    # Initialize the true consensus rankings, that is, those
    # that must be obtained from the rankings previously defined
    consensus_true = np.array([1, 1, 1])
    consensus_random_true = np.array([1, 1, 1])
    consensus_top_true = np.array([1, 1, 1])

    # Initialize the predicted consensus ranking, that is,
    # those obtained using the Bucket Pivot Algorithm
    # with multiple pivots and two-stages algorithm
    consensus_pred = bpa_lia_mp2.aggregate(Y)
    consensus_random_pred = bpa_lia_mp2.aggregate(Y_random)
    consensus_top_pred = bpa_lia_mp2.aggregate(Y_top)

    # Assert that the consensus rankings are the same
    np.testing.assert_array_equal(consensus_pred, consensus_true)
    np.testing.assert_array_equal(consensus_random_pred, consensus_random_true)
    np.testing.assert_array_equal(consensus_top_true, consensus_top_pred)


@pytest.mark.beta_invalid
def test_beta_invalid():
    """Test that invalid beta values raises errors."""
    # Assert that no errors are raised when using valid beta values
    RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2", beta=0)
    RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2", beta=0.5)
    RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2", beta=1)

    # Assert that the proper error is raised when using
    # a beta value that is not integer or floating type
    with pytest.raises(TypeError):
        RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2", beta="foo")

    # Assert that the proper error is raised
    # when using a beta value out of bounds
    with pytest.raises(ValueError):
        RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2", beta=-2)
    with pytest.raises(ValueError):
        RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2", beta=2)
