"""Naive implementation of the Bucket Pivot Algorithm, to test the Cython one.

It follows equation (3) of [alfaro_learning_2021], with weights, and equations (1)
and (2), Definition 1 and Figures 1, 2, 4 and 5 of [aledo_utopia_2017], in exact
arithmetic with :class:`fractions.Fraction`, with lists of items for the buckets and
the choices that the sources leave open made as in the Cython implementation.
Definition 1 rounds :math:`C(y_j, y_i)` into the entry of the labels :math:`y_i` and
:math:`y_j` of the utopian matrix, which its own Example 1 contradicts, so
:math:`C(y_i, y_j)` is rounded instead.
"""

import itertools
from fractions import Fraction

import numpy as np
from sklearn.utils import check_random_state

BETA = Fraction(1, 4)


def random_partial_rankings(n_samples, n_labels, *, missing, random_state):
    """Draw partial rankings, each label unranked with probability `missing`."""
    rng = check_random_state(random_state)
    y = rng.randint(1, n_labels + 1, size=(n_samples, n_labels)).astype(np.float64)
    y[rng.rand(n_samples, n_labels) < missing] = np.nan
    # Make the positions of the ranked labels dense from one again
    for row in y:
        is_ranked = ~np.isnan(row)
        row[is_ranked] = np.unique(row[is_ranked], return_inverse=True)[1] + 1
    return y


def pair_order_matrix(y, sample_weight):
    """Compute the pair order matrix of some rankings.

    The entries of the pairs that no ranking ranks are one half. The total weight
    of the rankings that rank each pair is returned too.
    """
    n_labels = len(y[0])
    matrix = [[Fraction(1, 2)] * n_labels for _ in range(n_labels)]
    totals = [[Fraction(0)] * n_labels for _ in range(n_labels)]
    for u, v in itertools.permutations(range(n_labels), 2):
        numerator = total = Fraction(0)
        for row, weight in zip(y, sample_weight):
            weight = Fraction(weight)
            if np.isnan(row[u]) or np.isnan(row[v]):
                continue
            # lambda(u, v) + lambda(v, u) is one for every ranking ranking both
            total += weight
            if row[u] < row[v]:
                numerator += weight
            elif row[u] == row[v]:
                numerator += weight / 2
        totals[u][v] = total
        if total:
            matrix[u][v] = numerator / total
    return matrix, totals


def utopian_matrix(matrix):
    """Compute the utopian matrix of a pair order matrix (Definition 1)."""

    def upsilon(x):
        if x > Fraction(3, 4):
            return Fraction(1)
        if x >= Fraction(1, 4):
            return Fraction(1, 2)
        return Fraction(0)

    return [[upsilon(x) for x in row] for row in matrix]


def utopia_value(matrix):
    """Compute the distance between a pair order matrix and its utopian matrix."""
    return distance(utopian_matrix(matrix), matrix)


def distance(bucket, matrix):
    """Compute the distance of equation (1) of [aledo_utopia_2017] of two matrices."""
    return sum(
        abs(b - c) for row_b, row_c in zip(bucket, matrix) for b, c in zip(row_b, row_c)
    )


def dani(matrix, items):
    """Compute the DANI index of each item among `items` (equation (2))."""
    utopian = utopian_matrix(matrix)
    return {
        u: sum(abs(matrix[u][v] - utopian[u][v]) for v in items if v != u)
        / (len(items) - 1)
        for u in items
    }


def bucket_matrix(buckets, n_labels):
    """Compute the bucket matrix of a bucket order given as a list of buckets."""
    position = {item: i for i, bucket in enumerate(buckets) for item in bucket}
    return [
        [
            Fraction(1)
            if position[u] < position[v]
            else Fraction(1, 2)
            if position[u] == position[v]
            else Fraction(0)
            for v in range(n_labels)
        ]
        for u in range(n_labels)
    ]


def lia_mp2(matrix, items, dani_index, *, pivot=None, multiple=True, second=True):
    """Order `items` into a list of buckets by the Bucket Pivot Algorithm.

    The pivot of each call has the least DANI in `dani_index`, computed once for
    all the items, as :math:`\\mathrm{LIA}_G` of Figure 2. `pivot` overrides the
    pivot of the first call, `multiple` uses the pivots of Figure 4 instead of the
    single one of Figure 1, and `second` adds the second stage of Figure 5.
    """
    if not items:
        return []
    if pivot is None:
        least = min(dani_index[u] for u in items)
        pivot = min(u for u in items if dani_index[u] == least)

    def side(pivots, u):
        mean = sum(matrix[i][u] for i in pivots) / len(pivots)
        if mean < Fraction(1, 2) - BETA:
            return "L"
        if mean <= Fraction(1, 2) + BETA:
            return "S"
        return "R"

    left, central, right = [], [pivot], []
    for u in items:
        if u != pivot:
            {"L": left, "S": central, "R": right}[
                side(central if multiple else [pivot], u)
            ].append(u)
    if second:
        pivots = list(central)
        rest, left, right = left + right, [], []
        for u in rest:
            {"L": left, "S": central, "R": right}[side(pivots, u)].append(u)
    return [
        *lia_mp2(matrix, sorted(left), dani_index, multiple=multiple, second=second),
        sorted(central),
        *lia_mp2(matrix, sorted(right), dani_index, multiple=multiple, second=second),
    ]


def aggregate(y, sample_weight=None):
    """Find the consensus bucket order of rankings, as positions of the labels.

    The labels with information are ordered by the Bucket Pivot Algorithm with the
    least indecision assumption, global DANI indices and two-stage multiple pivots,
    :math:`\\mathrm{LIA}_G^{\\mathrm{MP2}}` in [aledo_utopia_2017], and the others are
    tied in a last bucket.
    """
    n_labels = len(y[0])
    if sample_weight is None:
        sample_weight = [1] * len(y)
    matrix, totals = pair_order_matrix(y, sample_weight)
    items = [u for u in range(n_labels) if any(totals[u])]
    buckets = lia_mp2(matrix, items, dani(matrix, items))
    rest = [u for u in range(n_labels) if u not in items]
    if rest:
        buckets.append(rest)
    ranking = np.empty(n_labels, dtype=np.intp)
    for position, bucket in enumerate(buckets, start=1):
        ranking[bucket] = position
    return ranking
