"""Protocol of [cheng_label_2010] for IB-PL and IB-Mal."""

from functools import partial

from experiments.datasets import missing_labels_scenarios
from experiments.search import grid_search
from sklearn.model_selection import KFold, RepeatedKFold

from sklr.neighbors import KNeighborsLabelRanker

PAPER = {
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
    "scenarios": missing_labels_scenarios([0, 0.3, 0.6]),
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
            "search": partial(grid_search, grid={"n_neighbors": [5, 10, 15, 20]}),
        },
        # The paper disables the distance weights of IB-Mal for a fair
        # comparison with IB-PL
        "ib-mal": {
            "estimator": KNeighborsLabelRanker,
            "params": {"model": "mallows", "weights": "uniform"},
            "search": partial(grid_search, grid={"n_neighbors": [5, 10, 15, 20]}),
        },
    },
}
