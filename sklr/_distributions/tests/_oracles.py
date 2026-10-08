"""Naive implementations of the estimation of the models, to test the Cython ones.

The Mallows model follows Propositions 1 and 2 and Algorithm 1 of
[cheng_decision_2009] literally, in exact arithmetic with :class:`fractions.Fraction`,
so that ties are not decided by rounding. The spread is found in floating point
instead, from the probabilities of all the rankings rather than a formula for their
expected distance. The Plackett-Luce model follows the log-likelihood and equation (30)
of [hunter_mm_2004] term by term, and is also fitted by a general optimizer.
"""

import itertools
from fractions import Fraction

import numpy as np
from scipy.optimize import brentq, minimize
from sklearn.utils import check_random_state


def random_rankings(n_samples, n_labels, *, missing, random_state):
    """Draw rankings without ties, each label unranked with probability `missing`."""
    rng = check_random_state(random_state)
    y = np.argsort(rng.rand(n_samples, n_labels), axis=1).argsort(axis=1) + 1.0
    y[rng.rand(n_samples, n_labels) < missing] = np.nan
    # Make the positions of the ranked labels dense from one again
    for row in y:
        is_ranked = ~np.isnan(row)
        row[is_ranked] = np.argsort(np.argsort(row[is_ranked])) + 1
    return y


def kendall_distance(y, z):
    """Count the pairs of labels that two complete rankings order differently."""
    return sum(
        (y[i] - y[j]) * (z[i] - z[j]) < 0
        for i, j in itertools.combinations(range(len(y)), 2)
    )


def extensions(y):
    """Yield the complete rankings that keep the order of the ranked labels of `y`."""
    ranked = np.flatnonzero(~np.isnan(y))
    for permutation in itertools.permutations(range(1, len(y) + 1)):
        z = np.array(permutation, dtype=np.float64)
        if all(
            (y[i] < y[j]) == (z[i] < z[j]) for i, j in itertools.combinations(ranked, 2)
        ):
            yield z


def complete_ranking(y, center):
    """Extend a ranking by inserting each unranked label as in Proposition 1."""
    n_labels = len(y)
    ranked = [k for k in range(n_labels) if not np.isnan(y[k])]
    keys = {k: (y[k], 0) for k in ranked}
    for i in range(n_labels):
        if np.isnan(y[i]):
            costs = [
                sum(1 for k in ranked if y[k] <= j and center[k] > center[i])
                + sum(1 for k in ranked if y[k] > j and center[k] < center[i])
                for j in range(len(ranked) + 1)
            ]
            # The key j + 0.5 puts the label after the ranked label at position j
            # and before the one at j + 1
            keys[i] = (costs.index(min(costs)) + 0.5, center[i])
    completed = np.empty(n_labels)
    for position, label in enumerate(sorted(range(n_labels), key=keys.__getitem__)):
        completed[label] = position + 1
    return completed


def borda_scores(y, sample_weight):
    """Compute the generalized Borda count of the labels (Proposition 2) exactly."""
    n_labels = y.shape[1]
    scores = [Fraction(0)] * n_labels
    for row, weight in zip(y, sample_weight):
        n_ranked = int(np.sum(~np.isnan(row)))
        for label in range(n_labels):
            if np.isnan(row[label]):
                votes = Fraction(n_labels + 1, 2)
            else:
                votes = Fraction(
                    (n_ranked - int(row[label]) + 1) * (n_labels + 1), n_ranked + 1
                )
            scores[label] += Fraction(weight) * votes
    return scores


def rank_by_scores(scores):
    """Rank the labels by decreasing score, the smaller index first on ties."""
    order = sorted(range(len(scores)), key=lambda label: (-scores[label], label))
    ranking = np.empty(len(scores), dtype=np.intp)
    ranking[order] = np.arange(1, len(scores) + 1)
    return ranking


def estimate_center(y, sample_weight):
    """Estimate the center by Algorithm 1, without an iteration cap.

    Returns the center and the number of iterations, ``0`` if the rankings are
    complete.
    """
    center = rank_by_scores(borda_scores(y, sample_weight))
    if not np.any(np.isnan(y)):
        return center, 0
    n_iter = 0
    while True:
        n_iter += 1
        completed = np.array([complete_ranking(row, center) for row in y])
        previous = center
        center = rank_by_scores(borda_scores(completed, sample_weight))
        if np.array_equal(center, previous):
            return center, n_iter


def expected_distance(theta, n_labels):
    """Compute the expected Kendall distance to the center over all the rankings."""
    identity = np.arange(n_labels)
    distances = np.array(
        [
            kendall_distance(permutation, identity)
            for permutation in itertools.permutations(range(n_labels))
        ]
    )
    probabilities = np.exp(-theta * distances)
    return probabilities @ distances / probabilities.sum()


def estimate_spread(y, center, sample_weight):
    """Estimate the spread by completing the rankings and solving for the mean."""
    distances = [kendall_distance(complete_ranking(row, center), center) for row in y]
    weights = [Fraction(weight) for weight in sample_weight]
    mean_distance = float(
        sum(w * d for w, d in zip(weights, distances, strict=True)) / sum(weights)
    )
    n_labels = y.shape[1]
    if mean_distance == 0:
        return np.inf
    if mean_distance >= n_labels * (n_labels - 1) / 4:
        return 0.0
    upper = 1.0
    while expected_distance(upper, n_labels) >= mean_distance:
        upper *= 2
    return brentq(
        lambda theta: expected_distance(theta, n_labels) - mean_distance,
        0,
        upper,
        xtol=1e-300,
        rtol=1e-15,
    )


def ordered_labels(row):
    """List the ranked labels of a ranking from the first to the last."""
    ranked = np.flatnonzero(~np.isnan(row))
    return list(ranked[np.argsort(row[ranked])])


def plackett_luce_log_likelihood(y, sample_weight, parameters):
    """Compute the weighted log-likelihood of the rankings for the Plackett-Luce model.

    Each ranking adds the logarithm of its probability, the product over its
    positions but the last of the parameter of the label there divided by the sum of
    the parameters of that label and those after it.
    """
    log_likelihood = 0.0
    for row, weight in zip(y, sample_weight, strict=True):
        labels = ordered_labels(row)
        for i in range(len(labels) - 1):
            log_likelihood += weight * (
                np.log(parameters[labels[i]]) - np.log(parameters[labels[i:]].sum())
            )
    return log_likelihood


def estimate_plackett_luce(y, sample_weight, *, tol, max_iter):
    """Estimate the parameters by the MM algorithm of equation (30), term by term.

    Some ranking with a positive weight must have two labels. The labels that are
    never ranked above the last label of such a ranking get a zero parameter.
    Returns the parameters, which add up to one, the number of iterations and
    whether they converged.
    """
    n_labels = y.shape[1]
    rankings = [
        (ordered_labels(row), weight)
        for row, weight in zip(y, sample_weight, strict=True)
        if weight > 0
    ]
    wins = np.zeros(n_labels)
    for labels, weight in rankings:
        for label in labels[:-1]:
            wins[label] += weight
    parameters = np.full(n_labels, 1 / n_labels)
    for n_iter in range(1, max_iter + 1):
        denominators = np.zeros(n_labels)
        for t in range(n_labels):
            for labels, weight in rankings:
                for i in range(len(labels) - 1):
                    if t in labels[i:]:
                        denominators[t] += weight / parameters[labels[i:]].sum()
        updated = np.zeros(n_labels)
        has_wins = wins > 0
        updated[has_wins] = wins[has_wins] / denominators[has_wins]
        updated /= updated.sum()
        change = np.max(np.abs(updated - parameters))
        parameters = updated
        if change <= tol:
            return parameters, n_iter, True
    return parameters, max_iter, False


def maximize_plackett_luce(y, sample_weight):
    """Maximize the log-likelihood over the logarithms of the parameters.

    The logarithm of the parameter of the first label is fixed to zero, since the
    likelihood does not change when all the parameters are scaled. The parameters
    are returned scaled to add up to one.
    """

    def negative_log_likelihood(log_parameters):
        parameters = np.exp(np.r_[0, log_parameters])
        return -plackett_luce_log_likelihood(y, sample_weight, parameters)

    log_parameters = minimize(
        negative_log_likelihood,
        np.zeros(y.shape[1] - 1),
        method="BFGS",
        options={"gtol": 1e-10},
    ).x
    parameters = np.exp(np.r_[0, log_parameters])
    return parameters / parameters.sum()
