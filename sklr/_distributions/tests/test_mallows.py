import itertools
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning

from sklr._distributions import _mallows
from sklr._distributions.mallows import (
    estimate_center,
    estimate_center_batch,
    estimate_spread,
    estimate_spread_batch,
)
from sklr._distributions.tests import _oracles
from sklr._distributions.tests._oracles import extensions, random_rankings


def _estimate_center_outputs(y, sample_weight, *, max_iter=100):
    """Estimate the center of one group and return all its outputs."""
    center, n_iter, converged = _mallows.estimate_center_batch(
        np.ascontiguousarray(y, dtype=np.float64)[np.newaxis],
        np.ascontiguousarray(sample_weight, dtype=np.float64)[np.newaxis],
        max_iter,
        1,
    )
    return center[0], n_iter[0], converged[0]


def _assert_sorts_scores(ranking, scores):
    """Check that a ranking orders the labels by decreasing score.

    The order of the labels with equal scores is not checked, since floating point
    rounding may turn ties into small differences.
    """
    order = np.argsort(ranking)
    assert all(scores[i] >= scores[j] for i, j in itertools.pairwise(order))


@pytest.mark.parametrize(
    "y, expected",
    [
        # The totals are 7, 4 and 7, and the tie goes to the first label
        ([[1, 2, 3], [2, 3, 1], [2, 3, 1]], [1, 3, 2]),
        # The generalized totals are 6, 13/3 and 23/3, and the completed rankings
        # keep that center
        ([[1, 2, np.nan], [2, np.nan, 1], [2, 3, 1]], [2, 3, 1]),
    ],
)
def test_estimate_center_examples(y, expected):
    """Check the center of small examples computed by hand."""
    assert_array_equal(estimate_center(y), expected)


@pytest.mark.parametrize("n_labels", [2, 4, 7])
def test_estimate_center_complete(n_labels):
    """Check the center of complete rankings against the sum of their votes."""
    rng = np.random.RandomState(0)
    y = random_rankings(20, n_labels, missing=0, random_state=0)
    sample_weight = rng.randint(1, 5, size=len(y))
    # Integer votes and weights add up exactly, so the ties are real and go to
    # the smaller index, which is the order of appearance of rankdata
    scores = sample_weight @ (n_labels + 1 - y)
    expected = rankdata(-scores, method="ordinal")
    center, n_iter, converged = _estimate_center_outputs(y, sample_weight)
    assert_array_equal(center, expected)
    assert n_iter == 0
    assert converged


@pytest.mark.parametrize("n_labels", [3, 4, 5])
def test_estimate_center_initial(n_labels):
    """Check the first center against the expected votes of uniform extensions.

    The rankings are not completed when ``max_iter=0``, so the result is the
    first center, which ranks the labels by their expected votes when each ranking
    is extended uniformly at random (Proposition 2 of [cheng_decision_2009]).
    """
    rng = np.random.RandomState(0)
    y = random_rankings(6, n_labels, missing=0.5, random_state=0)
    sample_weight = rng.rand(len(y))
    scores = [Fraction(0)] * n_labels
    for row, weight in zip(y, sample_weight):
        candidates = list(extensions(row))
        for z in candidates:
            for label in range(n_labels):
                scores[label] += (
                    Fraction(weight) * int(n_labels + 1 - z[label]) / len(candidates)
                )
    center, _, _ = _estimate_center_outputs(y, sample_weight, max_iter=0)
    _assert_sorts_scores(center, scores)


@pytest.mark.parametrize("n_labels", [3, 5, 8])
@pytest.mark.parametrize("missing", [0, 0.3, 0.7])
def test_estimate_center_algorithm(n_labels, missing):
    """Check the center against a naive Algorithm 1 of [cheng_decision_2009]."""
    rng = np.random.RandomState(0)
    for _ in range(20):
        y = random_rankings(10, n_labels, missing=missing, random_state=rng)
        sample_weight = rng.rand(len(y))
        expected, expected_n_iter = _oracles.estimate_center(y, sample_weight)
        center, n_iter, converged = _estimate_center_outputs(y, sample_weight)
        assert_array_equal(center, expected)
        assert n_iter == expected_n_iter
        assert converged


def test_estimate_center_sample_weight():
    """Check the properties of the sample weights."""
    rng = np.random.RandomState(0)
    y = random_rankings(8, 6, missing=0.4, random_state=0)
    sample_weight = rng.randint(1, 4, size=len(y))
    expected = estimate_center(y, sample_weight)

    assert_array_equal(estimate_center(y), estimate_center(y, np.ones(len(y))))
    # Scaling by a power of two is exact, so not even the ties change
    assert_array_equal(estimate_center(y, 2.0 * sample_weight), expected)
    # A zero weight is the same as dropping the ranking
    sample_weight_zero = np.r_[sample_weight, 0]
    y_extra = np.vstack([y, random_rankings(1, 6, missing=0.4, random_state=1)])
    assert_array_equal(estimate_center(y_extra, sample_weight_zero), expected)


def test_estimate_center_sample_weight_repeat():
    """Check that an integer weight is the same as repeating the ranking.

    The votes of complete rankings are integers, so their totals are exact and the
    ties are the same. The fractional votes of incomplete rankings add up with
    rounding errors that depend on the order of the sums, which may break ties
    differently.
    """
    rng = np.random.RandomState(0)
    y = random_rankings(8, 6, missing=0, random_state=0)
    sample_weight = rng.randint(1, 4, size=len(y))
    assert_array_equal(
        estimate_center(np.repeat(y, sample_weight, axis=0)),
        estimate_center(y, sample_weight),
    )


def test_estimate_center_invariances():
    """Check that the center follows the labels and ignores the sample order."""
    rng = np.random.RandomState(0)
    y = random_rankings(10, 6, missing=0.4, random_state=0)
    # Random weights make ties between the totals of the labels unlikely, which
    # would otherwise be broken by their indices
    sample_weight = rng.rand(len(y))
    expected = estimate_center(y, sample_weight)

    labels = rng.permutation(y.shape[1])
    assert_array_equal(estimate_center(y[:, labels], sample_weight), expected[labels])
    samples = rng.permutation(len(y))
    assert_array_equal(estimate_center(y[samples], sample_weight[samples]), expected)


def test_estimate_center_single_ranking():
    """Check that the center of a single ranking is one of its extensions."""
    y = np.array([[np.nan, 2, np.nan, 1, np.nan]])
    center = estimate_center(y)
    assert any(np.array_equal(center, z) for z in extensions(y[0]))
    assert_array_equal(estimate_center([[3, 1, 2]]), [3, 1, 2])


def test_estimate_center_unranked():
    """Check that rankings with at most one ranked label give the label order."""
    y = [[np.nan, np.nan, np.nan], [1, np.nan, np.nan]]
    assert_array_equal(estimate_center(y[:1]), [1, 2, 3])
    assert_array_equal(estimate_center(y, [0, 1]), [1, 2, 3])


def test_estimate_center_input():
    """Check that integer positions are accepted and that the input is not modified."""
    y = random_rankings(5, 4, missing=0.3, random_state=0)
    y_copy = y.copy()
    center = estimate_center(y)
    assert_array_equal(y, y_copy)
    assert center.dtype == np.intp

    y = random_rankings(5, 4, missing=0, random_state=0)
    assert_array_equal(estimate_center(y.astype(np.int64)), estimate_center(y))


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_estimate_center_batch(n_threads):
    """Check that each group gets the center it gets alone, whatever the threads."""
    rng = np.random.RandomState(0)
    n_groups, n_samples, n_labels = 25, 7, 6
    y = random_rankings(n_groups * n_samples, n_labels, missing=0.4, random_state=0)
    y = y.reshape(n_groups, n_samples, n_labels)
    sample_weight = rng.rand(n_groups, n_samples)
    expected = [
        estimate_center(group, weights) for group, weights in zip(y, sample_weight)
    ]
    center = estimate_center_batch(y, sample_weight, n_threads=n_threads)
    assert_array_equal(center, expected)


def test_estimate_center_convergence_warning():
    """Check the warning when the center does not converge in time."""
    y = [[1, 2, np.nan], [2, np.nan, 1], [2, 3, 1]]
    with pytest.warns(ConvergenceWarning, match=r"1 of 1 .* max_iter=0"):
        estimate_center(y, max_iter=0)
    with pytest.warns(ConvergenceWarning, match=r"1 of 2 .* max_iter=0"):
        estimate_center_batch(
            np.array([y, [[1, 2, 3]] * 3], dtype=np.float64),
            np.ones((2, 3)),
            max_iter=0,
        )


def _rankings_at_mean_distance(n_labels, mean_distance):
    """Build the identity and its reverse, weighted to have a given mean distance.

    The identity is the center, and its reverse is at the largest distance from it,
    ``n_labels * (n_labels - 1) / 2``, so its weight is the fraction of that
    distance in the mean.
    """
    identity = np.arange(1.0, n_labels + 1)
    fraction = mean_distance / (n_labels * (n_labels - 1) / 2)
    return np.array([identity, identity[::-1]]), [1 - fraction, fraction], identity


@pytest.mark.parametrize("mean_distance", [1e-6, 0.1, 0.3, 0.49])
def test_estimate_spread_two_labels(mean_distance):
    """Check the spread of two labels against its closed form.

    With two labels, the expected distance is :math:`q / (1 + q)`, where :math:`q`
    is the exponential of minus the spread. The tolerance covers the rounding of
    the weights and of the expected distance.
    """
    y, sample_weight, center = _rankings_at_mean_distance(2, mean_distance)
    expected = np.log((1 - mean_distance) / mean_distance)
    assert_allclose(estimate_spread(y, center, sample_weight), expected, rtol=1e-12)


@pytest.mark.parametrize("n_labels", [3, 4, 5, 6])
@pytest.mark.parametrize("theta", [0.05, 0.5, 2, 6])
def test_estimate_spread_expected_distance(n_labels, theta):
    """Check that the spread is recovered from its expected distance.

    The expected distance is computed from the probabilities of all the rankings,
    which add up :math:`n!` terms, so the tolerance allows for their rounding.
    """
    mean_distance = _oracles.expected_distance(theta, n_labels)
    y, sample_weight, center = _rankings_at_mean_distance(n_labels, mean_distance)
    assert_allclose(estimate_spread(y, center, sample_weight), theta, rtol=1e-12)


@pytest.mark.parametrize("theta", [1e-8, 1e-4])
def test_estimate_spread_near_zero(theta):
    """Check the spread near zero against the expansion of the expected distance.

    The expected distance is the sum of the means of independent variables, uniform
    on :math:`0` to :math:`j - 1` when the spread is zero, so its derivative there
    is minus the sum of their variances, :math:`(j^2 - 1) / 12`. The third central
    moments are zero by symmetry, so the expansion is exact up to the cube of the
    spread. The tolerance covers the rounding of the expected distance of many
    labels, which is large next to its small difference from that of the uniform
    distribution.
    """
    n_labels = 50
    j = np.arange(1, n_labels + 1)
    mean_distance = n_labels * (n_labels - 1) / 4 - theta * np.sum(j**2 - 1) / 12
    y, sample_weight, center = _rankings_at_mean_distance(n_labels, mean_distance)
    assert_allclose(estimate_spread(y, center, sample_weight), theta, rtol=1e-6)


@pytest.mark.parametrize("theta", [1, 10, 100, 700])
def test_estimate_spread_large(theta):
    """Check large spreads against the closed form of the expected distance.

    The closed form of [cheng_decision_2009] only loses precision near zero, so it is
    an independent check of large spreads with many labels. A spread of ``700``
    keeps the weight of the reverse ranking, which is of the order of its
    exponential, above the smallest normal floating point number.
    """
    n_labels = 50
    q = np.exp(-theta)
    j = np.arange(1, n_labels + 1)
    mean_distance = n_labels * q / (1 - q) - np.sum(j * q**j / (1 - q**j))
    y, sample_weight, center = _rankings_at_mean_distance(n_labels, mean_distance)
    assert_allclose(estimate_spread(y, center, sample_weight), theta, rtol=1e-12)


@pytest.mark.parametrize("n_labels", [3, 4, 5])
@pytest.mark.parametrize("missing", [0, 0.4])
@pytest.mark.parametrize("estimated_center", [True, False])
def test_estimate_spread_oracle(n_labels, missing, estimated_center):
    """Check the spread against a naive estimation from the completed rankings.

    The center is the one of Algorithm 1 of [cheng_decision_2009] or a random one,
    which may be farther than the uniform distribution and give a zero spread. The
    tolerance covers the rounding of the expected distance of the oracle.
    """
    rng = np.random.RandomState(0)
    for _ in range(10):
        y = random_rankings(8, n_labels, missing=missing, random_state=rng)
        sample_weight = rng.rand(len(y))
        if estimated_center:
            center = estimate_center(y, sample_weight)
        else:
            center = rng.permutation(n_labels) + 1
        expected = _oracles.estimate_spread(y, center, sample_weight)
        assert_allclose(estimate_spread(y, center, sample_weight), expected, rtol=1e-12)


def test_estimate_spread_bounds():
    """Check the spreads of rankings that agree with the center or cancel out."""
    center = [1, 2, 3, 4]
    assert estimate_spread([center] * 3, center) == np.inf
    # The completed rankings are the center
    y = [[1, np.nan, 2, np.nan], [np.nan, 1, 2, 3], [np.nan] * 4]
    assert estimate_spread(y, center) == np.inf
    # A ranking and its reverse are at a mean distance of n(n-1)/4, that of the
    # uniform distribution, and a larger weight for the reverse moves it farther
    y = [center, center[::-1]]
    assert estimate_spread(y, center) == 0
    assert estimate_spread(y, center, [1, 2]) == 0


def test_estimate_spread_sample_weight():
    """Check the properties of the sample weights.

    The distances are integers, so the weighted sums with integer weights are
    exact and do not depend on the order of the sums.
    """
    rng = np.random.RandomState(0)
    y = random_rankings(8, 6, missing=0.4, random_state=0)
    center = estimate_center(y)
    sample_weight = rng.randint(1, 4, size=len(y))
    expected = estimate_spread(y, center, sample_weight)

    assert estimate_spread(y, center) == estimate_spread(y, center, np.ones(len(y)))
    assert estimate_spread(y, center, 2.0 * sample_weight) == expected
    assert estimate_spread(np.repeat(y, sample_weight, axis=0), center) == expected
    y_extra = np.vstack([y, random_rankings(1, 6, missing=0.4, random_state=1)])
    sample_weight_zero = np.r_[sample_weight, 0]
    assert estimate_spread(y_extra, center, sample_weight_zero) == expected


def test_estimate_spread_invariances():
    """Check that the spread ignores the order of the labels and of the samples.

    Moving the samples changes the order of the sums of the weighted distances, so
    the tolerance covers their rounding.
    """
    rng = np.random.RandomState(0)
    y = random_rankings(10, 6, missing=0.4, random_state=0)
    sample_weight = rng.rand(len(y))
    center = estimate_center(y, sample_weight)
    expected = estimate_spread(y, center, sample_weight)

    labels = rng.permutation(y.shape[1])
    assert estimate_spread(y[:, labels], center[labels], sample_weight) == expected
    samples = rng.permutation(len(y))
    assert_allclose(
        estimate_spread(y[samples], center, sample_weight[samples]),
        expected,
        rtol=1e-12,
    )


def test_estimate_spread_input():
    """Check that integer inputs are accepted and that the inputs are not modified."""
    y = random_rankings(5, 4, missing=0.3, random_state=0)
    center = np.array([2.0, 1, 4, 3])
    y_copy, center_copy = y.copy(), center.copy()
    expected = estimate_spread(y, center)
    assert_array_equal(y, y_copy)
    assert_array_equal(center, center_copy)
    assert estimate_spread(y, center.astype(np.int64)) == expected

    y = random_rankings(5, 4, missing=0, random_state=0)
    assert estimate_spread(y.astype(np.int64), center) == estimate_spread(y, center)


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_estimate_spread_batch(n_threads):
    """Check that each group gets the spread it gets alone, whatever the threads."""
    rng = np.random.RandomState(0)
    n_groups, n_samples, n_labels = 25, 7, 6
    y = random_rankings(n_groups * n_samples, n_labels, missing=0.4, random_state=0)
    y = y.reshape(n_groups, n_samples, n_labels)
    sample_weight = rng.rand(n_groups, n_samples)
    center = estimate_center_batch(y, sample_weight)
    expected = [
        estimate_spread(group, group_center, weights)
        for group, group_center, weights in zip(y, center, sample_weight)
    ]
    spread = estimate_spread_batch(y, center, sample_weight, n_threads=n_threads)
    assert_array_equal(spread, expected)
