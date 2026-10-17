import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse.csgraph import connected_components
from sklearn.exceptions import ConvergenceWarning

from sklr._distributions import _plackett_luce
from sklr._distributions.plackett_luce import (
    estimate_parameters,
    estimate_parameters_batch,
)
from sklr._distributions.tests import _oracles
from sklr._distributions.tests._oracles import ordered_labels, random_rankings


def _estimate_parameters_outputs(y, sample_weight, *, tol=1e-6, max_iter=1000):
    """Estimate the parameters of one group and return all its outputs."""
    parameters, n_iter, converged = _plackett_luce.estimate_parameters_batch(
        np.ascontiguousarray(y, dtype=np.float64)[np.newaxis],
        np.ascontiguousarray(sample_weight, dtype=np.float64)[np.newaxis],
        tol,
        max_iter,
        1,
    )
    return parameters[0], n_iter[0], converged[0]


def _has_maximum(y):
    """Tell whether the likelihood has a maximum (Assumption 1 of [hunter_mm_2004]).

    It has one when every label can be reached from every other one by following
    the labels that some ranking puts below them.
    """
    n_labels = y.shape[1]
    beats = np.zeros((n_labels, n_labels))
    for row in y:
        labels = ordered_labels(row)
        for i, label in enumerate(labels):
            beats[label, labels[i + 1 :]] = 1
    n_components, _ = connected_components(beats, connection="strong")
    return n_components == 1


@pytest.mark.parametrize(
    "y, sample_weight, expected",
    [
        # With two labels the parameters are proportional to the weighted wins, here
        # 3 and 1, and they are reached in one iteration from any start
        ([[1, 2], [1, 2], [2, 1], [1, 2]], None, [0.75, 0.25]),
        ([[1, 2], [2, 1]], [2.5, 0.5], [5 / 6, 1 / 6]),
    ],
)
def test_estimate_parameters_two_labels(y, sample_weight, expected):
    """Check the parameters of two labels against their closed form."""
    assert_allclose(estimate_parameters(y, sample_weight), expected, rtol=1e-15)


@pytest.mark.parametrize("n_labels", [3, 5, 8])
@pytest.mark.parametrize("missing", [0, 0.3, 0.6])
@pytest.mark.parametrize("max_iter", [1, 5, 1000])
def test_estimate_parameters_algorithm(n_labels, missing, max_iter):
    """Check the parameters against a naive equation (30) of [hunter_mm_2004].

    Both add up the same terms in a different order, so the tolerance covers their
    rounding, which grows a little at each iteration.
    """
    rng = np.random.RandomState(0)
    for _ in range(10):
        y = random_rankings(10, n_labels, missing=missing, random_state=rng)
        sample_weight = rng.rand(len(y))
        expected, expected_n_iter, expected_converged = _oracles.estimate_plackett_luce(
            y, sample_weight, tol=1e-6, max_iter=max_iter
        )
        parameters, n_iter, converged = _estimate_parameters_outputs(
            y, sample_weight, max_iter=max_iter
        )
        assert_allclose(parameters, expected, rtol=1e-10, atol=1e-300)
        assert n_iter == expected_n_iter
        assert converged == expected_converged


@pytest.mark.parametrize("n_labels", [3, 4, 6])
@pytest.mark.parametrize("missing", [0, 0.3])
def test_estimate_parameters_maximum_likelihood(n_labels, missing):
    """Check the parameters against a direct maximization of the likelihood.

    The rankings are drawn so that the maximum exists, where the MM algorithm
    converges. The small tolerance of the estimation leaves the error to the
    optimizer of the oracle.
    """
    rng = np.random.RandomState(0)
    for _ in range(5):
        y = random_rankings(15, n_labels, missing=missing, random_state=rng)
        assert _has_maximum(y)
        sample_weight = rng.rand(len(y)) + 0.5
        expected = _oracles.maximize_plackett_luce(y, sample_weight)
        parameters = estimate_parameters(y, sample_weight, tol=1e-12)
        assert_allclose(parameters, expected, rtol=1e-6)


def test_estimate_parameters_monotone():
    """Check that each iteration does not decrease the log-likelihood.

    This is the property of all the MM algorithms, so it holds whether the maximum
    exists or not. The tolerance covers the rounding of the log-likelihood.
    """
    rng = np.random.RandomState(0)
    y = random_rankings(12, 6, missing=0.4, random_state=0)
    sample_weight = rng.rand(len(y))
    log_likelihoods = [
        _oracles.plackett_luce_log_likelihood(
            y,
            sample_weight,
            _estimate_parameters_outputs(y, sample_weight, tol=0, max_iter=max_iter)[0],
        )
        for max_iter in range(30)
    ]
    assert np.all(np.diff(log_likelihoods) >= -1e-12)


def test_estimate_parameters_sum():
    """Check that the parameters are positive or zero and add up to one."""
    y = random_rankings(10, 7, missing=0.5, random_state=0)
    parameters = estimate_parameters(y)
    assert np.all(parameters >= 0)
    assert_allclose(parameters.sum(), 1, rtol=1e-14)


def test_estimate_parameters_no_wins():
    """Check that the labels never ranked above another one get a zero parameter.

    Labels 2 and 3 are always last, and label 4 is only ranked alone, so the
    likelihood does not depend on its parameter.
    """
    y = [
        [1, 2, 3, np.nan, np.nan],
        [2, 1, np.nan, 3, np.nan],
        [2, 1, 3, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, 1],
    ]
    parameters, _, converged = _estimate_parameters_outputs(y, np.ones(4))
    assert converged
    assert_array_equal(parameters[2:], 0)
    assert np.all(parameters[:2] > 0)


@pytest.mark.parametrize(
    "y, sample_weight",
    [
        ([[np.nan, np.nan, np.nan]], [1]),
        ([[1, np.nan, np.nan], [np.nan, np.nan, 1]], [1, 1]),
        # The only ranking with more than one label has a zero weight
        ([[1, 2, 3], [np.nan, 1, np.nan]], [0, 1]),
    ],
)
def test_estimate_parameters_no_information(y, sample_weight):
    """Check that the parameters are uniform when no ranking compares two labels."""
    parameters, n_iter, converged = _estimate_parameters_outputs(y, sample_weight)
    assert_array_equal(parameters, np.full(3, 1 / 3))
    assert n_iter == 0
    assert converged


@pytest.mark.parametrize("n_labels", [3, 5])
def test_estimate_parameters_agreeing_rankings(n_labels):
    """Check rankings that all agree, whose likelihood has no maximum.

    The parameter of the first label goes to one and the others to zero, each
    faster than the one before, so they keep the order of the rankings and a
    smaller tolerance takes them closer to that limit. The last label is never
    ranked above another one and gets a zero parameter.
    """
    y = np.tile(np.arange(1.0, n_labels + 1), (10, 1))
    parameters = estimate_parameters(y, max_iter=10_000)
    closer = estimate_parameters(y, tol=1e-9, max_iter=100_000)
    for estimate in (parameters, closer):
        assert np.all(np.diff(estimate[:-1]) < 0)
        assert estimate[-1] == 0
    assert 1 > closer[0] > parameters[0] > 0.99


def test_estimate_parameters_sample_weight():
    """Check the properties of the sample weights.

    Scaling by a power of two is exact. Repeating the rankings changes the order of
    the sums, which may stop the iterations at a different one, so a small
    tolerance, which the likelihood with a maximum reaches quickly, leaves the
    comparison to their rounding.
    """
    rng = np.random.RandomState(0)
    y = random_rankings(20, 6, missing=0.3, random_state=0)
    assert _has_maximum(y)
    sample_weight = rng.randint(1, 4, size=len(y))
    expected = estimate_parameters(y, sample_weight)

    assert_array_equal(estimate_parameters(y), estimate_parameters(y, np.ones(len(y))))
    assert_array_equal(estimate_parameters(y, 2.0 * sample_weight), expected)
    assert_allclose(
        estimate_parameters(np.repeat(y, sample_weight, axis=0), tol=1e-13),
        estimate_parameters(y, sample_weight, tol=1e-13),
        rtol=1e-10,
    )
    y_extra = np.vstack([y, random_rankings(1, 6, missing=0.4, random_state=1)])
    sample_weight_zero = np.r_[sample_weight, 0]
    assert_array_equal(estimate_parameters(y_extra, sample_weight_zero), expected)


@pytest.mark.parametrize("scale", [1e300, 1e-320])
def test_estimate_parameters_extreme_weights(scale):
    """Check that very large or very small weights give the same parameters.

    Their sums would overflow or lose most of their digits, but the algorithm does
    not change when all the weights are scaled.
    """
    rng = np.random.RandomState(0)
    y = random_rankings(20, 6, missing=0.3, random_state=0)
    sample_weight = rng.randint(1, 4, size=len(y)).astype(np.float64)
    assert_allclose(
        estimate_parameters(y, scale * sample_weight),
        estimate_parameters(y, sample_weight),
        rtol=1e-12,
    )


def test_estimate_parameters_invariances():
    """Check that the parameters follow the labels and ignore the sample order.

    Both change the order of the sums, so the tolerance covers their rounding.
    """
    rng = np.random.RandomState(0)
    y = random_rankings(10, 6, missing=0.4, random_state=0)
    sample_weight = rng.rand(len(y))
    expected = estimate_parameters(y, sample_weight)

    labels = rng.permutation(y.shape[1])
    assert_allclose(
        estimate_parameters(y[:, labels], sample_weight), expected[labels], rtol=1e-10
    )
    samples = rng.permutation(len(y))
    assert_allclose(
        estimate_parameters(y[samples], sample_weight[samples]), expected, rtol=1e-10
    )


def test_estimate_parameters_input():
    """Check that integer positions are accepted and that the input is not modified."""
    y = random_rankings(5, 4, missing=0.3, random_state=0)
    y_copy = y.copy()
    parameters = estimate_parameters(y)
    assert_array_equal(y, y_copy)
    assert parameters.dtype == np.float64

    y = random_rankings(5, 4, missing=0, random_state=0)
    assert_array_equal(estimate_parameters(y.astype(np.int64)), estimate_parameters(y))


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_estimate_parameters_batch(n_threads):
    """Check that each group gets the parameters it gets alone, whatever the threads.

    Some groups have no maximum of the likelihood, and the cap on the iterations
    leaves room for their slow convergence.
    """
    rng = np.random.RandomState(0)
    n_groups, n_samples, n_labels = 25, 7, 6
    y = random_rankings(n_groups * n_samples, n_labels, missing=0.4, random_state=0)
    y = y.reshape(n_groups, n_samples, n_labels)
    sample_weight = rng.rand(n_groups, n_samples)
    expected = [
        estimate_parameters(group, weights, max_iter=10_000)
        for group, weights in zip(y, sample_weight)
    ]
    parameters = estimate_parameters_batch(
        y, sample_weight, max_iter=10_000, n_threads=n_threads
    )
    assert_array_equal(parameters, expected)


def test_estimate_parameters_convergence_warning():
    """Check the warning when the parameters do not converge in time."""
    y = [[1, 2, 3], [2, 1, 3], [3, 1, 2]]
    with pytest.warns(ConvergenceWarning, match=r"1 of 1 .* tol=1e-06 .* max_iter=0"):
        parameters = estimate_parameters(y, max_iter=0)
    assert_array_equal(parameters, np.full(3, 1 / 3))
    with pytest.warns(ConvergenceWarning, match=r"1 of 2 .* max_iter=1"):
        estimate_parameters_batch(
            np.array([y, [[1, np.nan, np.nan]] * 3], dtype=np.float64),
            np.ones((2, 3)),
            max_iter=1,
        )
