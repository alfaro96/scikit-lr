import numpy as np
import pytest
from scipy.sparse import csr_array
from scipy.stats import rankdata

from sklr.utils import type_of_ranking

nan = np.nan
inf = np.inf


def _naive_type_of_ranking(y):
    """Classify `y` row by row from the dense ranks computed by SciPy."""
    is_label_ranking = True
    for row in y:
        ranked = row[~np.isnan(row)]
        # The ranked labels are densely positioned from one if and only if
        # their positions are the dense ranks of themselves
        if not np.array_equal(ranked, rankdata(ranked, method="dense")):
            return "unknown"
        is_label_ranking &= np.unique(ranked).size == ranked.size
    return "label_ranking" if is_label_ranking else "partial_label_ranking"


@pytest.mark.parametrize(
    "y, expected",
    [
        ([[1, 2, 3], [3, 1, 2]], "label_ranking"),
        ([[1, 2], [2, 1]], "label_ranking"),
        ([[1, 2, 3], [1, 1, 2]], "partial_label_ranking"),
        ([[1, 1, 1], [2, 1, 3]], "partial_label_ranking"),
        # Only one sample
        ([[2, 3, 1]], "label_ranking"),
        ([[2, 2, 1]], "partial_label_ranking"),
        # Positions that are not dense or do not start at one
        ([[1, 1, 3], [1, 2, 3]], "unknown"),
        ([[1, 2, 4], [1, 2, 3]], "unknown"),
        ([[2, 3, 4], [1, 2, 3]], "unknown"),
        ([[0, 1, 2], [1, 2, 3]], "unknown"),
        ([[-1, 1, 2], [1, 2, 3]], "unknown"),
        # Positions that are not integers
        ([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]], "label_ranking"),
        ([[1.0, 1.5, 2.0], [1.0, 2.0, 3.0]], "unknown"),
        ([[1.5, 2.5, 3.5], [1.0, 2.0, 3.0]], "unknown"),
    ],
)
def test_type_of_ranking(y, expected):
    assert type_of_ranking(y) == expected


@pytest.mark.parametrize(
    "y, expected",
    [
        # The positions of the ranked labels are dense from one
        ([[2, nan, 1], [1, 2, 3]], "label_ranking"),
        ([[nan, 1, 1], [1, 2, 3]], "partial_label_ranking"),
        ([[2, nan, 3], [1, 2, 3]], "unknown"),
        ([[nan, 2, 2], [1, 2, 3]], "unknown"),
        # A row with zero or one ranked label is a valid incomplete ranking
        ([[1, nan, nan], [1, 2, 3]], "label_ranking"),
        ([[nan, nan, nan], [1, 2, 3]], "label_ranking"),
        ([[nan, nan, nan], [1, 1, 2]], "partial_label_ranking"),
        ([[nan, nan], [nan, nan]], "label_ranking"),
        ([[nan, 2, nan], [1, 2, 3]], "unknown"),
        # NaN is the only marker of an unranked label
        ([[1, 2, inf], [1, 2, 3]], "unknown"),
        ([[1, 2, -inf], [1, 2, 3]], "unknown"),
    ],
)
def test_type_of_ranking_incomplete(y, expected):
    assert type_of_ranking(y) == expected


@pytest.mark.parametrize(
    "y",
    [
        # Not two-dimensional
        [1, 2, 3],
        [[[1, 2], [2, 1]]],
        1,
        # Without samples or with less than two labels
        np.empty((0, 3)),
        [[1], [1]],
        np.empty((2, 0)),
        # Without numeric positions
        [["a", "b"], ["b", "a"]],
        [[True, True], [True, True]],
        [[1 + 0j, 2 + 0j], [2 + 0j, 1 + 0j]],
        # Ragged
        [[1, 2], [1, 2, 3]],
        # Sparse
        csr_array([[1, 2], [2, 1]]),
    ],
)
def test_type_of_ranking_invalid_input(y):
    assert type_of_ranking(y) == "unknown"


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int32, np.int64, np.uint8, np.uint64, np.float32, np.float64],
)
def test_type_of_ranking_dtype(dtype):
    assert type_of_ranking(np.array([[1, 2, 3], [3, 1, 2]], dtype=dtype)) == (
        "label_ranking"
    )
    assert type_of_ranking(np.array([[1, 1, 2], [3, 1, 2]], dtype=dtype)) == (
        "partial_label_ranking"
    )
    assert type_of_ranking(np.array([[1, 1, 3], [3, 1, 2]], dtype=dtype)) == ("unknown")


def test_type_of_ranking_object_dtype():
    y = np.array([[1, 2, 3], [3, 1, 2]], dtype=object)
    assert type_of_ranking(y) == "label_ranking"


def test_type_of_ranking_does_not_modify_input():
    y = np.array([[3.0, nan, 1.0], [1.0, 1.0, 2.0]])
    y_copy = y.copy()
    type_of_ranking(y)
    np.testing.assert_array_equal(y, y_copy)


@pytest.mark.parametrize("n_labels", [2, 3, 5, 8])
@pytest.mark.parametrize("seed", range(5))
def test_type_of_ranking_naive_oracle(n_labels, seed):
    rng = np.random.default_rng(seed)
    n_samples = 100
    # Valid rankings by construction: the dense ranks of the ranked labels from
    # random scores, which tie when they repeat
    scores = rng.integers(0, n_labels, size=(n_samples, n_labels)).astype(float)
    scores[rng.random(scores.shape) < 0.2] = nan
    y = rankdata(scores, method="dense", axis=1, nan_policy="omit")
    # Moving a label one position down keeps some rankings valid and breaks
    # the density of others
    rows = np.flatnonzero(rng.random(n_samples) < 0.5)
    y[rows, rng.integers(0, n_labels, size=rows.size)] += 1

    types = [type_of_ranking(row[np.newaxis]) for row in y]
    assert types == [_naive_type_of_ranking(row[np.newaxis]) for row in y]
    # With these seeds, each type is generated
    assert set(types) == {"label_ranking", "partial_label_ranking", "unknown"}
    valid = [type_ != "unknown" for type_ in types]
    assert type_of_ranking(y[valid]) == _naive_type_of_ranking(y[valid])
    assert type_of_ranking(y) == "unknown"
