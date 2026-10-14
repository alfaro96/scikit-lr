import os
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklr._aggregation.bucket_pivot import aggregate, aggregate_batch
from sklr._aggregation.tests import _oracles
from sklr._aggregation.tests._oracles import random_partial_rankings

# The permutations 1|2|3, 1|3|2 and 2|1|3 of Example 1 of [aledo_utopia_2017], as
# the positions of the labels
EXAMPLE_RANKINGS = [[1, 2, 3], [1, 3, 2], [2, 1, 3]]

# The partial rankings of Table 4 of [alfaro_learning_2021], of the labels CS, Ma,
# Ph, Ch and Bi
TABLE_RANKINGS = [
    [1, 2, 3, 3, 3],
    [np.nan, np.nan, 1, 1, 1],
    [1, 2, np.nan, 3, 3],
    [np.nan, np.nan, 1, 1, np.nan],
    [1, 2, 3, 3, 3],
]


def test_oracle_examples():
    """Check the oracle against Examples 1, 3 and 4 of [aledo_utopia_2017]."""
    matrix, _ = _oracles.pair_order_matrix(np.array(EXAMPLE_RANKINGS), [1, 1, 1])
    assert matrix == [
        [Fraction(1, 2), Fraction(2, 3), Fraction(1)],
        [Fraction(1, 3), Fraction(1, 2), Fraction(2, 3)],
        [Fraction(0), Fraction(1, 3), Fraction(1, 2)],
    ]
    assert _oracles.utopian_matrix(matrix) == [
        [Fraction(1, 2), Fraction(1, 2), Fraction(1)],
        [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
        [Fraction(0), Fraction(1, 2), Fraction(1, 2)],
    ]
    largest = Fraction(1, 4) * 3 * 2
    assert _oracles.utopia_value(matrix) == Fraction(2, 3)
    assert (largest - _oracles.utopia_value(matrix)) / largest == Fraction(5, 9)
    dani = _oracles.dani(matrix, [0, 1, 2])
    assert dani == {0: Fraction(1, 12), 1: Fraction(1, 6), 2: Fraction(1, 12)}

    # Example 3: the single pivot of Figure 1 with each item as the pivot
    expected = {0: [[0, 1], [2]], 1: [[0, 1, 2]], 2: [[0], [1, 2]]}
    for pivot, buckets in expected.items():
        result = _oracles.lia_mp2(
            matrix, [0, 1, 2], dani, pivot=pivot, multiple=False, second=False
        )
        assert result == buckets
    objective = {
        pivot: _oracles.distance(_oracles.bucket_matrix(buckets, 3), matrix)
        for pivot, buckets in expected.items()
    }
    assert objective == {0: 1, 1: Fraction(5, 3), 2: 1}

    # Example 4: the multiple pivots of Figure 4
    expected = {0: [[0, 1], [2]], 1: [[0, 1], [2]], 2: [[0], [1, 2]]}
    for pivot, buckets in expected.items():
        result = _oracles.lia_mp2(matrix, [0, 1, 2], dani, pivot=pivot, second=False)
        assert result == buckets


def test_oracle_pair_order_matrix():
    """Check the oracle against Example 3.1 of [alfaro_learning_2021]."""
    matrix, _ = _oracles.pair_order_matrix(np.array(TABLE_RANKINGS), [1] * 5)
    # CS precedes Ma, both precede the others wherever they are ranked, and the
    # others are tied
    for u in range(5):
        for v in range(u + 1, 5):
            assert matrix[u][v] == (Fraction(1) if u < 2 else Fraction(1, 2))


@pytest.mark.parametrize(
    "y, expected",
    [
        # The least DANI is that of the first and the last labels, so the first one
        # is the pivot and the second label joins it, as in Example 4 of
        # [aledo_utopia_2017]
        (EXAMPLE_RANKINGS, [1, 1, 2]),
        # The rankings of Example 3.1 of [alfaro_learning_2021], whose pair order
        # matrix is the bucket matrix of CS|Ma|Ph,Ch,Bi and hence its own optimum
        (TABLE_RANKINGS, [1, 2, 3, 3, 3]),
        # A pair order entry of exactly 1/4 or 3/4 ties the labels, and 1/5 does not
        ([[1, 2]] + [[2, 1]] * 3, [1, 1]),
        ([[1, 2]] * 3 + [[2, 1]], [1, 1]),
        ([[1, 2]] + [[2, 1]] * 4, [2, 1]),
        # A label without information does not drag the others into its bucket
        ([[1, 2, 3, np.nan]] * 3, [1, 2, 3, 4]),
        ([[np.nan, 1, 2, 3]] * 3, [4, 1, 2, 3]),
        # A single label, and labels tied by every ranking
        ([[1]], [1]),
        ([[1, 1, 1]] * 2, [1, 1, 1]),
        # Rankings of a single label give no information
        ([[1, np.nan, np.nan], [np.nan, 1, np.nan]], [1, 1, 1]),
        ([[np.nan, np.nan, np.nan]], [1, 1, 1]),
    ],
)
def test_aggregate_examples(y, expected):
    """Check the consensus of small examples computed by hand."""
    assert_array_equal(aggregate(np.array(y, dtype=np.float64)), expected)


@pytest.mark.parametrize("n_samples", [1, 2, 5, 16])
@pytest.mark.parametrize("n_labels", [2, 3, 5, 8])
def test_aggregate_oracle_complete(n_samples, n_labels):
    """Check the consensus of complete partial rankings against the oracle."""
    rng = np.random.RandomState(0)
    for seed in range(20):
        y = random_partial_rankings(n_samples, n_labels, missing=0, random_state=seed)
        sample_weight = rng.randint(1, 4, size=n_samples)
        assert_array_equal(
            aggregate(y, sample_weight), _oracles.aggregate(y, sample_weight)
        )


@pytest.mark.parametrize("missing", [0.3, 0.6])
@pytest.mark.parametrize("n_labels", [3, 5, 8])
def test_aggregate_oracle_incomplete(missing, n_labels):
    """Check the consensus of incomplete partial rankings against the oracle."""
    for seed in range(50):
        y = random_partial_rankings(10, n_labels, missing=missing, random_state=seed)
        assert_array_equal(aggregate(y), _oracles.aggregate(y))


@pytest.mark.parametrize("n_labels", [2, 4, 7])
def test_aggregate_bucket_order(n_labels):
    """Check that the consensus of copies of a bucket order is that bucket order."""
    for seed in range(20):
        y = random_partial_rankings(1, n_labels, missing=0, random_state=seed)
        assert_array_equal(aggregate(np.repeat(y, 3, axis=0)), y[0])


def test_aggregate_sample_weight():
    """Check that integer weights repeat the rankings and zero weights drop them.

    Scaling the weights does not change the consensus either.
    """
    y = random_partial_rankings(12, 5, missing=0.3, random_state=0)
    sample_weight = np.random.RandomState(0).randint(0, 4, size=len(y))
    expected = aggregate(np.repeat(y, sample_weight, axis=0))
    assert_array_equal(aggregate(y, sample_weight), expected)
    assert_array_equal(aggregate(y, 0.1 * sample_weight), expected)
    assert_array_equal(aggregate(y), aggregate(y, np.ones(len(y))))


@pytest.mark.parametrize("n_threads", [1, 2, 3])
def test_aggregate_batch(n_threads):
    """Check that the batch version aggregates each group on its own."""
    y = np.stack(
        [
            random_partial_rankings(7, 6, missing=0.3, random_state=seed)
            for seed in range(5)
        ]
    )
    sample_weight = np.random.RandomState(0).rand(5, 7)
    expected = [aggregate(group, weight) for group, weight in zip(y, sample_weight)]
    assert_array_equal(aggregate_batch(y, sample_weight, n_threads=n_threads), expected)


def test_aggregate_many_labels():
    """Check that many labels do not overflow a small stack of the threads.

    A recursion as deep as the number of labels, one per bucket, overflows a stack
    of 256 KiB with a thousand labels.
    """
    code = """
import numpy as np
from sklr._aggregation.bucket_pivot import aggregate_batch
y = np.arange(1.0, 1001)[np.newaxis]
result = aggregate_batch(np.stack([y, y[:, ::-1]]), np.ones((2, 1)), n_threads=2)
assert (result == np.vstack([y, y[:, ::-1]])).all()
"""
    # The size of the stack of the threads is only read when OpenMP starts
    env = {**os.environ, "OMP_STACKSIZE": "256K"}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
