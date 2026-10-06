import itertools

import numpy as np
import pytest
from scipy.stats import kendalltau

from sklr.metrics import kendall_distance, kendall_tau_score, kendall_tau_x_score

METRICS = {
    "kendall_tau_score": kendall_tau_score,
    "kendall_tau_x_score": kendall_tau_x_score,
    "kendall_distance": kendall_distance,
    "unnormalized_kendall_distance": lambda *args, **kwargs: kendall_distance(
        *args, normalize=False, **kwargs
    ),
}

# The metrics that reject rankings with ties
LABEL_RANKING_METRICS = [
    "kendall_tau_score",
    "kendall_distance",
    "unnormalized_kendall_distance",
]


def _linear_orderings(n_labels):
    """Yield all the rankings of the labels without ties."""
    for permutation in itertools.permutations(range(1, n_labels + 1)):
        yield np.array(permutation)


def _weak_orderings(n_labels):
    """Yield all the rankings of the labels, with and without ties."""
    for positions in itertools.product(range(1, n_labels + 1), repeat=n_labels):
        # The positions are dense from one if they are all the integers from one
        # to the greatest of them
        if set(positions) == set(range(1, max(positions) + 1)):
            yield np.array(positions)


def _random_rankings(n_samples, n_labels, *, ties, random_state):
    """Draw random complete rankings, with ties if `ties` is ``True``."""
    rng = np.random.RandomState(random_state)
    if not ties:
        return np.argsort(rng.rand(n_samples, n_labels), axis=1).argsort(axis=1) + 1
    # Dense positions of random integers, which are often tied
    values = rng.randint(n_labels, size=(n_samples, n_labels))
    return np.array([np.unique(row, return_inverse=True)[1] + 1 for row in values])


def _kendall_score_matrix(y):
    """Kendall's score matrix of a ranking, with ``0`` for tied labels.

    Emond and Mason (2002), section 3.2.
    """
    n_labels = len(y)
    a = np.zeros((n_labels, n_labels))
    for i, j in itertools.product(range(n_labels), repeat=2):
        if y[i] < y[j]:
            a[i, j] = 1
        elif y[i] > y[j]:
            a[i, j] = -1
    return a


def _tau_x_score_matrix(y):
    """Emond and Mason's score matrix of a ranking, with ``1`` for tied labels.

    Emond and Mason (2002), section 4.
    """
    n_labels = len(y)
    a = np.zeros((n_labels, n_labels))
    for i, j in itertools.product(range(n_labels), repeat=2):
        if i != j:
            a[i, j] = 1 if y[i] <= y[j] else -1
    return a


def _naive_kendall_tau(y_true, y_pred):
    """Kendall's tau of two rankings without ties, scoring the pairs one by one.

    Kendall (1938), equation (1): twice the total score of the pairs, +1 for a
    concordant pair and -1 for a discordant one, divided by n(n - 1).
    """
    n_labels = len(y_true)
    score = 0
    for i, j in itertools.combinations(range(n_labels), 2):
        score += 1 if (y_true[i] < y_true[j]) == (y_pred[i] < y_pred[j]) else -1
    return 2 * score / (n_labels * (n_labels - 1))


def _naive_kendall_tau_x(y_true, y_pred):
    """Emond and Mason's tau_x of two rankings, from their score matrices.

    Emond and Mason (2002), equation (3).
    """
    n_labels = len(y_true)
    a, b = _tau_x_score_matrix(y_true), _tau_x_score_matrix(y_pred)
    return np.sum(a * b) / (n_labels * (n_labels - 1))


def _naive_kendall_distance(y_true, y_pred):
    """Number of discordant pairs of two rankings without ties."""
    return sum(
        (y_true[i] < y_true[j]) != (y_pred[i] < y_pred[j])
        for i, j in itertools.combinations(range(len(y_true)), 2)
    )


def _kemeny_snell_distance(y_true, y_pred):
    """Kemeny-Snell distance of two rankings, from Kendall's score matrices.

    Emond and Mason (2002), equation (2).
    """
    a, b = _kendall_score_matrix(y_true), _kendall_score_matrix(y_pred)
    return np.sum(np.abs(a - b)) / 2


@pytest.mark.parametrize("n_labels", [2, 3, 4])
def test_kendall_tau_score_all_linear_orderings(n_labels):
    for y_true, y_pred in itertools.product(_linear_orderings(n_labels), repeat=2):
        expected = _naive_kendall_tau(y_true, y_pred)
        assert kendall_tau_score([y_true], [y_pred]) == pytest.approx(expected)
        # Without ties, Kendall's tau is the tau-b that SciPy computes by default
        assert expected == pytest.approx(kendalltau(y_true, y_pred).statistic)


@pytest.mark.parametrize("n_labels", [2, 3, 4])
def test_kendall_tau_x_score_all_weak_orderings(n_labels):
    for y_true, y_pred in itertools.product(_weak_orderings(n_labels), repeat=2):
        expected = _naive_kendall_tau_x(y_true, y_pred)
        assert kendall_tau_x_score([y_true], [y_pred]) == pytest.approx(expected)
        # Emond and Mason (2002), equation (A3): tau_x and the Kemeny-Snell
        # distance are equivalent
        distance = _kemeny_snell_distance(y_true, y_pred)
        assert expected == pytest.approx(1 - 2 * distance / (n_labels * (n_labels - 1)))


@pytest.mark.parametrize("n_labels", [2, 3, 4])
def test_kendall_distance_all_linear_orderings(n_labels):
    n_pairs = n_labels * (n_labels - 1) / 2
    for y_true, y_pred in itertools.product(_linear_orderings(n_labels), repeat=2):
        expected = _naive_kendall_distance(y_true, y_pred)
        assert kendall_distance([y_true], [y_pred], normalize=False) == expected
        assert kendall_distance([y_true], [y_pred]) == pytest.approx(expected / n_pairs)
        # Two half-flips make one swap of adjacent labels (Emond and Mason,
        # 2002, section 5), so the Kemeny-Snell distance between rankings
        # without ties is twice the Kendall distance
        assert _kemeny_snell_distance(y_true, y_pred) == 2 * expected


def test_kendall_tau_score_kendall_example():
    # Kendall (1938), sections 2 and 5: the total score of the pairs is 5 out of
    # a maximum of 45
    y_true = [[6, 9, 4, 3, 5, 10, 2, 1, 8, 7]]
    y_pred = [[6, 5, 10, 2, 3, 9, 7, 4, 1, 8]]
    assert kendall_tau_score(y_true, y_pred) == pytest.approx(5 / 45)
    # Kendall (1938), section 4: 25 of the pairs are concordant, so 20 of them
    # are discordant
    assert kendall_distance(y_true, y_pred, normalize=False) == 20


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        # A ranking with all the labels tied has a coefficient of 1 with itself,
        # unlike Kendall's tau-b (Emond and Mason, 2002, sections 3.3 and 4),
        # and of 0 with any ranking without ties
        ([1, 1, 1], [1, 1, 1], 1),
        ([1, 1, 1], [1, 2, 3], 0),
        ([1, 1, 1], [3, 1, 2], 0),
        # The rankings only disagree on the ordered pair (1, 0): the second
        # label is tied with the first one in the true ranking, but behind it in
        # the predicted one. That gives (5 - 1) / 6
        ([1, 1, 2], [1, 2, 3], 2 / 3),
        ([1, 2, 3], [3, 2, 1], -1),
        # Only rankings without ties reach -1: the tied labels of a ranking and
        # of its reverse agree on their pair in both orders
        ([1, 1, 2], [2, 2, 1], -1 / 3),
    ],
)
def test_kendall_tau_x_score_ties(y_true, y_pred, expected):
    assert kendall_tau_x_score([y_true], [y_pred]) == pytest.approx(expected)


@pytest.mark.parametrize("n_labels", [2, 5, 10])
def test_kendall_tau_x_score_equals_kendall_tau_score_without_ties(n_labels):
    y_true = _random_rankings(50, n_labels, ties=False, random_state=0)
    y_pred = _random_rankings(50, n_labels, ties=False, random_state=1)
    assert kendall_tau_x_score(y_true, y_pred) == pytest.approx(
        kendall_tau_score(y_true, y_pred)
    )


@pytest.mark.parametrize("n_labels", [2, 5, 10])
def test_kendall_distance_kendall_tau_score_relation(n_labels):
    y_true = _random_rankings(50, n_labels, ties=False, random_state=0)
    y_pred = _random_rankings(50, n_labels, ties=False, random_state=1)
    assert kendall_distance(y_true, y_pred) == pytest.approx(
        (1 - kendall_tau_score(y_true, y_pred)) / 2
    )


@pytest.mark.parametrize("name", METRICS)
def test_average_of_samples(name):
    metric = METRICS[name]
    ties = name not in LABEL_RANKING_METRICS
    y_true = _random_rankings(20, 5, ties=ties, random_state=0)
    y_pred = _random_rankings(20, 5, ties=ties, random_state=1)
    expected = np.mean([metric([t], [p]) for t, p in zip(y_true, y_pred)])
    assert metric(y_true, y_pred) == pytest.approx(expected)


@pytest.mark.parametrize("name", METRICS)
def test_bounds(name):
    metric = METRICS[name]
    ties = name not in LABEL_RANKING_METRICS
    y_true = _random_rankings(10, 4, ties=ties, random_state=0)
    reverse = y_true.max(axis=1, keepdims=True) + 1 - y_true
    best, worst = {
        "kendall_tau_score": (1, -1),
        "kendall_tau_x_score": (1, -1),
        "kendall_distance": (0, 1),
        "unnormalized_kendall_distance": (0, 6),
    }[name]
    assert metric(y_true, y_true) == pytest.approx(best)
    if not ties:
        assert metric(y_true, reverse) == pytest.approx(worst)


@pytest.mark.parametrize("name", METRICS)
def test_symmetry(name):
    metric = METRICS[name]
    ties = name not in LABEL_RANKING_METRICS
    y_true = _random_rankings(20, 5, ties=ties, random_state=0)
    y_pred = _random_rankings(20, 5, ties=ties, random_state=1)
    assert metric(y_true, y_pred) == pytest.approx(metric(y_pred, y_true))


@pytest.mark.parametrize("name", METRICS)
def test_invariance_to_permutations(name):
    metric = METRICS[name]
    ties = name not in LABEL_RANKING_METRICS
    y_true = _random_rankings(20, 5, ties=ties, random_state=0)
    y_pred = _random_rankings(20, 5, ties=ties, random_state=1)
    sample_weight = np.random.RandomState(2).randint(1, 5, size=20)
    expected = metric(y_true, y_pred, sample_weight=sample_weight)
    rng = np.random.RandomState(3)
    # The order of the samples does not matter, and neither does the order of
    # the labels if both rankings are permuted in the same way (Kemeny-Snell
    # axiom 2, Emond and Mason, 2002, table I)
    samples, labels = rng.permutation(20), rng.permutation(5)
    assert metric(
        y_true[samples], y_pred[samples], sample_weight=sample_weight[samples]
    ) == pytest.approx(expected)
    assert metric(
        y_true[:, labels], y_pred[:, labels], sample_weight=sample_weight
    ) == pytest.approx(expected)


@pytest.mark.parametrize("name", METRICS)
def test_sample_weight(name):
    metric = METRICS[name]
    ties = name not in LABEL_RANKING_METRICS
    y_true = _random_rankings(20, 5, ties=ties, random_state=0)
    y_pred = _random_rankings(20, 5, ties=ties, random_state=1)
    sample_weight = np.random.RandomState(2).randint(1, 5, size=20)

    unweighted = metric(y_true, y_pred)
    weighted = metric(y_true, y_pred, sample_weight=sample_weight)
    assert metric(y_true, y_pred, sample_weight=np.ones(20)) == pytest.approx(
        unweighted
    )
    assert weighted != pytest.approx(unweighted)
    # Integer weights are the same as repeating the samples
    repeated = np.repeat(np.arange(20), sample_weight)
    assert metric(y_true[repeated], y_pred[repeated]) == pytest.approx(weighted)
    # Scaling the weights does not change the average
    assert metric(y_true, y_pred, sample_weight=sample_weight * 0.1) == pytest.approx(
        weighted
    )
    # Samples with zero weight are the same as leaving them out
    sample_weight_zeros = sample_weight.copy()
    sample_weight_zeros[::2] = 0
    assert metric(y_true, y_pred, sample_weight=sample_weight_zeros) == pytest.approx(
        metric(y_true[1::2], y_pred[1::2], sample_weight=sample_weight[1::2])
    )
    # A list is the same as an array
    assert metric(
        y_true.tolist(), y_pred.tolist(), sample_weight=sample_weight.tolist()
    ) == pytest.approx(weighted)


@pytest.mark.parametrize("name", METRICS)
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint8, np.float32, np.float64]
)
def test_dtypes(name, dtype):
    metric = METRICS[name]
    ties = name not in LABEL_RANKING_METRICS
    y_true = _random_rankings(20, 5, ties=ties, random_state=0)
    y_pred = _random_rankings(20, 5, ties=ties, random_state=1)
    expected = metric(y_true, y_pred)
    assert metric(y_true.astype(dtype), y_pred.astype(dtype)) == pytest.approx(expected)


@pytest.mark.parametrize("name", METRICS)
def test_returns_python_float(name):
    metric = METRICS[name]
    assert type(metric([[1, 2]], [[2, 1]])) is float


@pytest.mark.parametrize("name", METRICS)
def test_single_sample_two_labels(name):
    metric = METRICS[name]
    expected = {
        "kendall_tau_score": (1, -1),
        "kendall_tau_x_score": (1, -1),
        "kendall_distance": (0, 1),
        "unnormalized_kendall_distance": (0, 1),
    }[name]
    assert metric([[1, 2]], [[1, 2]]) == expected[0]
    assert metric([[1, 2]], [[2, 1]]) == expected[1]


@pytest.mark.parametrize("name", METRICS)
@pytest.mark.parametrize("input_name", ["y_true", "y_pred"])
def test_incomplete_rankings_rejected(name, input_name):
    metric = METRICS[name]
    rankings = {"y_true": np.array([[1, 2, 3]]), "y_pred": np.array([[1, 2, 3]])}
    rankings[input_name] = np.array([[1, np.nan, 2]])
    msg = f"Expected complete rankings in {input_name}"
    with pytest.raises(ValueError, match=msg):
        metric(**rankings)


@pytest.mark.parametrize("name", LABEL_RANKING_METRICS)
@pytest.mark.parametrize("input_name", ["y_true", "y_pred"])
def test_ties_rejected(name, input_name):
    metric = METRICS[name]
    rankings = {"y_true": np.array([[1, 2, 3]]), "y_pred": np.array([[1, 2, 3]])}
    rankings[input_name] = np.array([[1, 1, 2]])
    with pytest.raises(
        ValueError, match=f"Expected rankings without ties in {input_name}"
    ):
        metric(**rankings)


@pytest.mark.parametrize("name", METRICS)
@pytest.mark.parametrize(
    "y_true, y_pred, msg",
    [
        (
            [[1, 2, 3], [1, 2, 3]],
            [[1, 2, 3]],
            "Found input variables with inconsistent numbers of samples",
        ),
        (
            [[1, 2, 3]],
            [[1, 2]],
            r"y_true and y_pred have different number of labels \(3 != 2\)",
        ),
        ([1, 2, 3], [[1, 2, 3]], "Expected a 2D array of shape"),
        ([[1, 3, 2]], [[1, 2, 4]], "Expected the positions of the ranked labels"),
        ([[1]], [[1]], "while a minimum of 2 is required"),
        ([[1, 2, np.inf]], [[1, 2, 3]], "Input y_true contains infinity"),
    ],
)
def test_invalid_rankings(name, y_true, y_pred, msg):
    metric = METRICS[name]
    with pytest.raises(ValueError, match=msg):
        metric(y_true, y_pred)


@pytest.mark.parametrize("name", METRICS)
@pytest.mark.parametrize(
    "sample_weight, msg",
    [
        ([1, -1], "Negative values in data passed to `sample_weight`"),
        ([0, 0], "Sample weights must contain at least one non-zero number"),
        ([1, 1, 1], "Found input variables with inconsistent numbers of samples"),
        ([[1], [1]], "Sample weights must be 1D array or scalar"),
    ],
)
def test_invalid_sample_weight(name, sample_weight, msg):
    metric = METRICS[name]
    with pytest.raises(ValueError, match=msg):
        metric([[1, 2], [2, 1]], [[1, 2], [1, 2]], sample_weight=sample_weight)


@pytest.mark.parametrize(
    "metric", [kendall_tau_score, kendall_tau_x_score, kendall_distance]
)
def test_invalid_params(metric):
    with pytest.raises(ValueError, match="The 'y_true' parameter"):
        metric("rankings", [[1, 2]])


def test_kendall_distance_invalid_normalize():
    with pytest.raises(ValueError, match="The 'normalize' parameter"):
        kendall_distance([[1, 2]], [[1, 2]], normalize="yes")
