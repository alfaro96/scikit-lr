"""Datasets, scenarios, metrics, cross-validation and methods of each paper.

Each paper in ``PAPERS`` declares the tag of its datasets in OpenML and their names
there, and its scenarios, each with a description and the function that transforms
the training rankings in it, which receives them and a random state, or ``None`` to
train with them as they are. It also declares the name of its metric in
``METRICS``, and its cross-validation: the outer splitter, whose repetitions, if it
has several, are run as separate jobs, and the inner splitter that chooses the
parameters of the methods in each outer training fold.
Finally, it declares the estimator of each method, its fixed parameters, those that
depend on the training data, if any, as a function that receives the samples and
rankings of each training fold and returns them, and the grid of those chosen by the
inner cross-validation, which is empty if none is chosen.
"""

from functools import partial

import numpy as np
from scipy.stats import rankdata
from sklearn.datasets import fetch_openml
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RepeatedKFold

from sklr.metrics import kendall_tau_score
from sklr.neighbors import KNeighborsLabelRanker

# The datasets uploaded to OpenML by the authors of scikit-lr, with their identifiers
# under each of their tags, since a name may have several versions there
OPENML_DATASETS = {
    "label ranking": {
        "authorship": 42834,
        "bodyfat": 42837,
        "calhousing": 42839,
        "cold": 42840,
        "cpu": 42841,
        "diau": 42842,
        "dtt": 42843,
        "elevators": 42845,
        "fried": 42846,
        "glass": 42847,
        "heat": 42849,
        "housing": 42850,
        "iris": 42851,
        "pendigits": 42856,
        "segment": 42859,
        "spo": 42861,
        "stock": 42862,
        "vehicle": 42863,
        "vowel": 42865,
        "wine": 42867,
        "wisconsin": 42869,
    },
    "partial label ranking": {
        "authorship": 42835,
        "blocks": 42836,
        "breast": 42838,
        "ecoli": 42844,
        "glass": 42848,
        "letter": 42853,
        "libras": 42855,
        "pendigits": 42857,
        "satimage": 42858,
        "segment": 42860,
        "vehicle": 42864,
        "vowel": 42866,
        "yeast": 42870,
        "iris": 42871,
        "wine": 42872,
    },
}


def load_dataset(tag, name, data_home=None):
    """Load the features and the rankings of a dataset from OpenML."""
    # The liac-arff parser avoids depending on pandas
    bunch = fetch_openml(
        data_id=OPENML_DATASETS[tag][name],
        as_frame=False,
        parser="liac-arff",
        data_home=data_home,
    )
    return bunch.data, np.asarray(bunch.target, dtype=np.float64)


def remove_labels(y, random_state, *, probability):
    """Remove each label of each ranking with a probability.

    The positions of the labels left are made dense from one again, keeping their
    ties. The rankings may be left with fewer than two labels, which are kept.
    """
    y = y.copy()
    y[random_state.rand(*y.shape) < probability] = np.nan
    return rankdata(y, axis=1, method="dense", nan_policy="omit")


def observed_kendall_tau(y_true, y_pred):
    """Average Kendall's tau over the pairs of labels ranked by each true ranking.

    The training rankings may be incomplete, and so are the validation rankings of
    the inner cross-validation, which the metrics of scikit-lr reject. The rankings
    with fewer than two ranked labels are left out.
    """
    scores = []
    for true, pred in zip(y_true, y_pred, strict=True):
        ranked = ~np.isnan(true)
        if ranked.sum() < 2:
            continue
        true, pred = true[ranked], pred[ranked]
        concordance = np.sign(true[:, None] - true) * np.sign(pred[:, None] - pred)
        n_pairs = ranked.sum() * (ranked.sum() - 1)
        scores.append(concordance.sum() / n_pairs)
    return np.mean(scores) if scores else 0.0


# The metric of each paper, with its name in LaTeX for the report and the figures,
# the function that scores the test rankings and the scorer of the inner
# cross-validation, whose validation rankings may be incomplete
METRICS = {
    "kendall_tau": {
        "latex": r"\text{Kendall's }\tau",
        "score": kendall_tau_score,
        "inner_scorer": make_scorer(observed_kendall_tau),
    },
}

PAPERS = {
    "cheng_decision_2009": {
        "datasets": {
            "tag": "label ranking",
            "names": [
                "authorship",
                "bodyfat",
                "calhousing",
                "cpu",
                "elevators",
                "fried",
                "glass",
                "housing",
                "iris",
                "pendigits",
                "segment",
                "stock",
                "vehicle",
                "vowel",
                "wine",
                "wisconsin",
            ],
        },
        "scenarios": {
            "complete": {
                "description": "complete training rankings",
                "transform": None,
            },
            "missing30": {
                "description": "30% of the training labels removed",
                "transform": partial(remove_labels, probability=0.3),
            },
            "missing60": {
                "description": "60% of the training labels removed",
                "transform": partial(remove_labels, probability=0.6),
            },
        },
        "metric": "kendall_tau",
        # Five repetitions of a ten-fold cross-validation (p. 7). The paper does not
        # say how many folds the inner cross-validation has, so it has five, and the
        # splits have fixed seeds
        "cross_validation": {
            "outer": RepeatedKFold(n_splits=10, n_repeats=5, random_state=0),
            "inner": KFold(n_splits=5, shuffle=True, random_state=0),
        },
        # The paper gives no grid of numbers of neighbors, so that of
        # [cheng_label_2010] (p. 5) is used
        "methods": {
            "iblr": {
                "estimator": KNeighborsLabelRanker,
                "params": {"model": "mallows", "weights": "dudani"},
                "grid": {"n_neighbors": [5, 10, 15, 20]},
            },
        },
    },
    "cheng_label_2010": {
        "datasets": {
            "tag": "label ranking",
            "names": [
                "authorship",
                "bodyfat",
                "calhousing",
                "cpu",
                "elevators",
                "fried",
                "glass",
                "housing",
                "iris",
                "pendigits",
                "segment",
                "stock",
                "vehicle",
                "vowel",
                "wine",
                "wisconsin",
            ],
        },
        "scenarios": {
            "complete": {
                "description": "complete training rankings",
                "transform": None,
            },
            "missing30": {
                "description": "30% of the training labels removed",
                "transform": partial(remove_labels, probability=0.3),
            },
            "missing60": {
                "description": "60% of the training labels removed",
                "transform": partial(remove_labels, probability=0.6),
            },
        },
        "metric": "kendall_tau",
        # Five repetitions of a ten-fold cross-validation (p. 6), and an inner one with
        # five folds and the same seeds as for [cheng_decision_2009]
        "cross_validation": {
            "outer": RepeatedKFold(n_splits=10, n_repeats=5, random_state=0),
            "inner": KFold(n_splits=5, shuffle=True, random_state=0),
        },
        # The grid of numbers of neighbors is on p. 5
        "methods": {
            "ib-pl": {
                "estimator": KNeighborsLabelRanker,
                "params": {"model": "plackett_luce", "weights": "uniform"},
                "grid": {"n_neighbors": [5, 10, 15, 20]},
            },
            # The paper disables the distance weights of IB-Mal for a fair
            # comparison with IB-PL
            "ib-mal": {
                "estimator": KNeighborsLabelRanker,
                "params": {"model": "mallows", "weights": "uniform"},
                "grid": {"n_neighbors": [5, 10, 15, 20]},
            },
        },
    },
}


def n_repetitions(splitter):
    """Count the repetitions of a splitter, one if it does not repeat its splits."""
    return getattr(splitter, "n_repeats", 1)


def make_ranker(paper, method, n_jobs):
    """Build the ranker of a method of a paper, with its fixed parameters.

    `n_jobs` is passed to the rankers that have that parameter.
    """
    method = PAPERS[paper]["methods"][method]
    ranker = method["estimator"](**method["params"])
    if "n_jobs" in ranker.get_params():
        ranker.set_params(n_jobs=n_jobs)
    return ranker
