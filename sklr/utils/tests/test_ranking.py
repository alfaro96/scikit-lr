import re

import numpy as np
import pytest
from scipy.sparse import csr_array, csr_matrix
from scipy.stats import rankdata
from sklearn.base import BaseEstimator

from sklr.base import LabelRankerMixin
from sklr.utils import check_ranking, type_of_ranking
from sklr.utils._ranking import _validate_data

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


@pytest.mark.parametrize(
    "y, allow_ties, allow_incomplete",
    [
        ([[1, 2, 3], [3, 1, 2]], False, False),
        ([[1, 2], [2, 1]], False, False),
        ([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]], False, False),
        ([[2, 3, 1]], False, False),
        ([[1, 1, 2], [1, 2, 3]], True, False),
        ([[1, 1, 1], [2, 1, 3]], True, False),
        ([[2, nan, 1], [1, 2, 3]], False, True),
        ([[nan, 1, 1], [1, 2, 3]], True, True),
        # A row with zero or one ranked label is a valid incomplete ranking
        ([[1, nan, nan], [1, 2, 3]], False, True),
        ([[nan, nan, nan], [1, 2, 3]], False, True),
        ([[nan, nan], [nan, nan]], False, True),
    ],
)
def test_check_ranking(y, allow_ties, allow_incomplete):
    y_converted = check_ranking(
        y, allow_ties=allow_ties, allow_incomplete=allow_incomplete
    )
    assert isinstance(y_converted, np.ndarray)
    assert y_converted.dtype == np.float64
    np.testing.assert_array_equal(y_converted, np.asarray(y, dtype=np.float64))


@pytest.mark.parametrize(
    "y, err_msg",
    [
        # Not dense or not starting at one
        ([[1, 2, 3], [1, 1, 3]], "dense from 1, got y[1] = [1 1 3]."),
        ([[1, 2, 4], [1, 2, 3]], "dense from 1, got y[0] = [1 2 4]."),
        ([[2, 3, 4], [1, 2, 3]], "dense from 1, got y[0] = [2 3 4]."),
        ([[0, 1, 2], [1, 2, 3]], "dense from 1, got y[0] = [0 1 2]."),
        ([[-1, 1, 2], [1, 2, 3]], "dense from 1, got y[0] = [-1  1  2]."),
        # Not integers
        ([[1.0, 1.5, 2.0], [1, 2, 3]], "dense from 1, got y[0] = [1.  1.5 2. ]."),
        # The ranked labels of incomplete rankings are not dense from one
        ([[2, nan, 3], [1, 2, 3]], "dense from 1, got y[0] = [ 2. nan  3.]."),
        ([[nan, 2, nan], [1, 2, 3]], "dense from 1, got y[0] = [nan  2. nan]."),
    ],
)
@pytest.mark.parametrize("allow_ties", [False, True])
def test_check_ranking_not_dense(y, err_msg, allow_ties):
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        check_ranking(y, allow_ties=allow_ties, allow_incomplete=True)


@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_check_ranking_ties(allow_incomplete):
    y = [[1, 2, 3], [3, 1, 2], [1, 1, 2], [1, 1, 1]]
    err_msg = "Expected rankings without ties in y, got y[2] = [1 1 2]."
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        check_ranking(y, allow_incomplete=allow_incomplete)
    np.testing.assert_array_equal(
        check_ranking(y, allow_ties=True, allow_incomplete=allow_incomplete), y
    )


@pytest.mark.parametrize("allow_ties", [False, True])
def test_check_ranking_incomplete(allow_ties):
    y = [[1, 2, 3], [nan, 1, 2], [nan, nan, nan]]
    err_msg = (
        "Expected complete rankings in y, without unranked labels (NaN), got "
        "y[1] = [nan  1.  2.]."
    )
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        check_ranking(y, allow_ties=allow_ties)
    np.testing.assert_array_equal(
        check_ranking(y, allow_ties=allow_ties, allow_incomplete=True), y
    )


@pytest.mark.parametrize("value", [inf, -inf])
@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_check_ranking_infinite(value, allow_incomplete):
    # NaN is the only marker of an unranked label
    with pytest.raises(ValueError, match="Input y contains infinity"):
        check_ranking([[1, 2, value]], allow_incomplete=allow_incomplete)


@pytest.mark.parametrize(
    "y, err_msg",
    [
        (1, "for y, got a 0D array instead."),
        ([1, 2, 3], "for y, got a 1D array instead."),
        ([[[1, 2], [2, 1]]], "for y, got a 3D array instead."),
        (
            np.empty((0, 3)),
            "Found y with 0 sample(s) (shape=(0, 3)) while a minimum of 1 is required.",
        ),
        (
            [[1], [1]],
            "Found y with 1 label(s) (shape=(2, 1)) while a minimum of 2 is required.",
        ),
        (
            np.empty((2, 0)),
            "Found y with 0 label(s) (shape=(2, 0)) while a minimum of 2 is required.",
        ),
        (
            [[True, False], [False, True]],
            "got dtype bool instead.",
        ),
        ([["a", "b"], ["b", "a"]], "dtype='numeric' is not compatible"),
        ([[1 + 0j, 2 + 0j], [2 + 0j, 1 + 0j]], "Complex data not supported"),
        ([[1, 2], [1, 2, 3]], "inhomogeneous shape"),
    ],
)
def test_check_ranking_invalid_input(y, err_msg):
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        check_ranking(y, allow_ties=True, allow_incomplete=True)


@pytest.mark.parametrize("csr_container", [csr_array, csr_matrix])
def test_check_ranking_sparse(csr_container):
    with pytest.raises(TypeError, match="Sparse data was passed for y"):
        check_ranking(csr_container([[1, 2], [2, 1]]))


@pytest.mark.parametrize(
    "y, kwargs, err_msg",
    [
        ([[1, 1, 3]], {}, "in y_true to be dense from 1, got y_true[0] = [1 1 3]."),
        ([[1, 1, 2]], {}, "without ties in y_true, got y_true[0] = [1 1 2]."),
        ([[1, nan, 2]], {}, "in y_true, without unranked labels (NaN), got y_true[0]"),
        ([[1, inf]], {}, "Input y_true contains infinity"),
        ([1, 2], {}, "for y_true, got a 1D array instead."),
        ([[1], [1]], {}, "Found y_true with 1 label(s)"),
        ([[True, False]], {}, "labels in y_true to be integer"),
    ],
)
def test_check_ranking_input_name(y, kwargs, err_msg):
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        check_ranking(y, input_name="y_true", **kwargs)


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int32, np.int64, np.uint8, np.uint64, np.float32, np.float64],
)
def test_check_ranking_dtype(dtype):
    y = np.array([[1, 2, 3], [3, 1, 2]], dtype=dtype)
    y_converted = check_ranking(y)
    assert y_converted.dtype == np.float64
    np.testing.assert_array_equal(y_converted, y)


def test_check_ranking_object_dtype():
    y = np.array([[1, 2, 3], [3, 1, 2]], dtype=object)
    y_converted = check_ranking(y)
    assert y_converted.dtype == np.float64
    np.testing.assert_array_equal(y_converted, y.astype(np.float64))


def test_check_ranking_does_not_modify_input():
    y = np.array([[2.0, nan, 1.0], [1.0, 1.0, 2.0]])
    y_copy = y.copy()
    check_ranking(y, allow_ties=True, allow_incomplete=True)
    np.testing.assert_array_equal(y, y_copy)


@pytest.mark.parametrize("n_labels", [2, 3, 5, 8])
@pytest.mark.parametrize("seed", range(5))
def test_check_ranking_naive_oracle(n_labels, seed):
    rng = np.random.default_rng(seed)
    n_samples = 100
    # The same generation as in test_type_of_ranking_naive_oracle
    scores = rng.integers(0, n_labels, size=(n_samples, n_labels)).astype(float)
    scores[rng.random(scores.shape) < 0.2] = nan
    y = rankdata(scores, method="dense", axis=1, nan_policy="omit")
    rows = np.flatnonzero(rng.random(n_samples) < 0.5)
    y[rows, rng.integers(0, n_labels, size=rows.size)] += 1

    for row in y[:, np.newaxis]:
        expected = _naive_type_of_ranking(row)
        # A ranking is accepted if and only if it is valid, and rejected
        # because of its ties if and only if it has them
        if expected == "unknown":
            with pytest.raises(ValueError, match="dense from 1"):
                check_ranking(row, allow_ties=True, allow_incomplete=True)
        else:
            check_ranking(row, allow_ties=True, allow_incomplete=True)
            if expected == "partial_label_ranking":
                with pytest.raises(ValueError, match="without ties"):
                    check_ranking(row, allow_incomplete=True)
            else:
                check_ranking(row, allow_incomplete=True)


@pytest.mark.parametrize(
    "y",
    [
        [[1, 2, 3], [3, 1, 2]],
        [[1, 1, 2], [3, 1, 2]],
        [[2, nan, 1], [3, 1, 2]],
        [[1, 1, 3], [3, 1, 2]],
        [[1, 2, inf], [1, 2, 3]],
        [[1], [1]],
        [1, 2, 3],
        [[True, True], [True, True]],
    ],
)
def test_check_ranking_consistent_with_type_of_ranking(y):
    # The rankings accepted with ties and incomplete rankings are the ones
    # whose type is known, and without ties, the label rankings
    type_ = type_of_ranking(y)
    for allow_ties, accepted_types in [
        (True, {"label_ranking", "partial_label_ranking"}),
        (False, {"label_ranking"}),
    ]:
        if type_ in accepted_types:
            check_ranking(y, allow_ties=allow_ties, allow_incomplete=True)
        else:
            with pytest.raises(ValueError):
                check_ranking(y, allow_ties=allow_ties, allow_incomplete=True)


class LabelRanker(LabelRankerMixin, BaseEstimator):
    pass


def test_validate_data():
    X = [[0, 1], [2, 3], [4, 5]]
    y = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    estimator = LabelRanker()
    X_validated, y_validated = _validate_data(estimator, X, y)
    np.testing.assert_array_equal(X_validated, X)
    np.testing.assert_array_equal(y_validated, y)
    assert y_validated.dtype == np.float64
    assert estimator.n_features_in_ == 2


def test_validate_data_y_none():
    err_msg = "requires y to be passed, but the target y is None"
    with pytest.raises(ValueError, match=err_msg):
        _validate_data(LabelRanker(), [[0, 1], [2, 3]], None)


def test_validate_data_inconsistent_length():
    err_msg = "Found input variables with inconsistent numbers of samples: [2, 1]"
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        _validate_data(LabelRanker(), [[0, 1], [2, 3]], [[1, 2, 3]])


def test_validate_data_rankings():
    X = [[0, 1], [2, 3]]
    y = [[1, 1, 2], [np.nan, 1, 2]]
    # Incomplete rankings are accepted by default, but not ties
    with pytest.raises(ValueError, match="Expected rankings without ties in y"):
        _validate_data(LabelRanker(), X, y)
    with pytest.raises(ValueError, match="Expected complete rankings in y"):
        _validate_data(LabelRanker(), X, y, allow_ties=True, allow_incomplete=False)
    _, y_validated = _validate_data(LabelRanker(), X, y, allow_ties=True)
    np.testing.assert_array_equal(y_validated, y)


def test_validate_data_check_params():
    X = [[0, np.nan], [2, 3]]
    y = [[1, 2], [2, 1]]
    with pytest.raises(ValueError, match="Input X contains NaN"):
        _validate_data(LabelRanker(), X, y)
    X_validated, _ = _validate_data(LabelRanker(), X, y, ensure_all_finite="allow-nan")
    np.testing.assert_array_equal(X_validated, X)
