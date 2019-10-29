"""Testing for consensus classes."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from plr.consensus import RankAggregationAlgorithm


# =============================================================================
# Initialization
# =============================================================================

# The rankings
Y = np.array([[1, 2, 3], [2, 3, 1], [2, 3, 1]])
Y_random = np.array([[1, 2, np.nan], [2, np.nan, 1], [2, 3, 1]])
Y_top = np.array([[1, 2, np.inf], [2, np.inf, 1], [2, 3, 1]])


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.parametrize("algorithm", ["foo", "borda_count", "bpa_lia_mp2"])
@pytest.mark.get_algorithm
def test_get_algorithm(algorithm):
    """Test the get_algorithm method from RankAggregationAlgorithm."""
    # Assert that an exception is raised for the "foo" algorithm
    if algorithm == "foo":
        with pytest.raises(ValueError):
            RankAggregationAlgorithm.get_algorithm(algorithm)
    # Otherwise, assert that a RankAggregationAlgorithm object is returned
    else:
        assert issubclass(
            type(RankAggregationAlgorithm.get_algorithm(algorithm)),
            RankAggregationAlgorithm)


@pytest.mark.aggregate_borda_count
def test_aggregate_borda_count():
    """Test the aggregate method for BordaCountAlgorithm."""
    # Initialize the BordaCountAlgorithm object to be tested
    borda_count = RankAggregationAlgorithm.get_algorithm("borda_count")

    # Assert that the consensus ranking obtained from complete ones is correct
    np.testing.assert_array_equal(
        borda_count.aggregate(Y),
        np.array([1, 3, 2]))

    # Assert that the consensus ranking obtained
    # from randomly missed items is correct
    np.testing.assert_array_equal(
        borda_count.aggregate(Y_random),
        np.array([2, 3, 1]))

    # Assert that the consensus ranking obtained from top-k items is correct
    np.testing.assert_array_equal(
        borda_count.aggregate(Y_top),
        np.array([1, 3, 2]))


@pytest.mark.aggregate_bpa_lia_mp2
def test_aggregate_bpa_lia_mp2():
    """Test the aggregate method for BPALIAMP2Algorithm."""
    # Initialize the BPALIAMP2Algorithm object to be tested
    bpa_lia_mp2 = RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2")

    # Assert that the consensus ranking obtained from complete ones is correct
    np.testing.assert_array_equal(
        bpa_lia_mp2.aggregate(Y),
        np.array([1, 2, 1]))

    # Assert that the consensus ranking obtained
    # from randomly missed items is correct
    np.testing.assert_array_equal(
        bpa_lia_mp2.aggregate(Y_random),
        np.array([1, 1, 1]))

    # Assert that the consensus ranking obtained from top-k items is correct
    np.testing.assert_array_equal(
        bpa_lia_mp2.aggregate(Y_top),
        np.array([1, 2, 1]))


@pytest.mark.sample_weight_invalid
@pytest.mark.parametrize("algorithm", ["borda_count", "bpa_lia_mp2"])
def test_sample_weight_invalid(algorithm):
    """Test that invalid sample weighting raises errors."""
    # Assert that an error is raised when the
    # sample weights is not a 1-D array
    with pytest.raises(ValueError):
        sample_weight = np.array(0)
        RankAggregationAlgorithm.get_algorithm(algorithm).aggregate(
            Y, sample_weight)

    with pytest.raises(ValueError):
        sample_weight = np.ones(Y.shape[0])[:, None]
        RankAggregationAlgorithm.get_algorithm(algorithm).aggregate(
            Y, sample_weight)

    # Assert that an error is raised when the
    # sample weights have wrong number of samples
    with pytest.raises(ValueError):
        sample_weight = np.ones(Y.shape[0] - 1)
        RankAggregationAlgorithm.get_algorithm(algorithm).aggregate(
            Y, sample_weight)

    with pytest.raises(ValueError):
        sample_weight = np.ones(Y.shape[0] + 1)
        RankAggregationAlgorithm.get_algorithm(algorithm).aggregate(
            Y, sample_weight)
