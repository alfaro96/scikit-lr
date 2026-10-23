"""Protocol of [cheng_decision_2009] for IBLR and LRT."""

from functools import partial

from experiments.datasets import missing_labels_scenarios
from experiments.search import grid_search
from sklearn.model_selection import KFold, RepeatedKFold

from sklr.neighbors import KNeighborsLabelRanker
from sklr.tree import DecisionTreeLabelRanker

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
    "metrics": ["kendall_tau"],
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
            "search": partial(grid_search, grid={"n_neighbors": [5, 10, 15, 20]}),
        },
        # The paper tunes no parameter of the tree, which stops growing on
        # consistent nodes and on nodes with fewer than 2|Y| labels, where Y is
        # the set of labels (p. 5), read as fewer than 2|Y| samples
        "lrt": {
            "estimator": DecisionTreeLabelRanker,
            "params": {"random_state": 0},
            "data_params": lambda X, y: {"min_samples_split": 2 * y.shape[1]},
        },
    },
}
