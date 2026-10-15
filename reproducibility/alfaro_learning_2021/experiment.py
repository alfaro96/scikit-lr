"""Protocol of [alfaro_learning_2021] for IBPLR and PLRT."""

from functools import partial

from experiments.datasets import missing_labels_scenarios
from experiments.search import doubling_search, inner_train_size
from sklearn.model_selection import KFold, RepeatedKFold

from sklr.neighbors import KNeighborsPartialLabelRanker
from sklr.tree import DecisionTreePartialLabelRanker

PAPER = {
    "datasets": {
        "tag": "partial label ranking",
        "names": [
            "authorship",
            "blocks",
            "breast",
            "ecoli",
            "glass",
            "iris",
            "letter",
            "libras",
            "pendigits",
            "satimage",
            "segment",
            "vehicle",
            "vowel",
            "wine",
            "yeast",
        ],
    },
    # Complete training rankings and 30% and 60% of their labels removed, as in
    # [cheng_decision_2009] (p. 907)
    "scenarios": missing_labels_scenarios([0, 0.3, 0.6]),
    "metric": "tau_x",
    # Five repetitions of a ten-fold cross-validation (p. 906) and an inner
    # five-fold one (p. 905), with the same seeds as for the other papers
    "cross_validation": {
        "outer": RepeatedKFold(n_splits=10, n_repeats=5, random_state=0),
        "inner": KFold(n_splits=5, shuffle=True, random_state=0),
    },
    # KNN-E-D in the tables, with the Euclidean distance and the inverse
    # distance weights, and the number of neighbors chosen from five by doubling it
    # and then by a binary search (p. 905). The paper does not bound it, so it is
    # bounded by the training samples of the inner folds
    "methods": {
        "ibplr": {
            "estimator": KNeighborsPartialLabelRanker,
            "params": {"weights": "distance"},
            "search": partial(
                doubling_search,
                param="n_neighbors",
                start=5,
                upper=inner_train_size,
            ),
        },
        # PLRT-A, PLRT-D, PLRT-E and PLRT-G in the tables, with the four criteria
        # and no tuned parameter. The trees stop growing on consistent nodes and,
        # as the label ranking trees, on nodes with fewer than 2n samples, where n
        # is the number of labels (pp. 901 and 897)
        **{
            method: {
                "estimator": DecisionTreePartialLabelRanker,
                "params": {"criterion": criterion, "random_state": 0},
                "data_params": lambda X, y: {"min_samples_split": 2 * y.shape[1]},
            }
            for method, criterion in [
                ("plrt_a", "disagreements"),
                ("plrt_d", "distance"),
                ("plrt_e", "entropy"),
                ("plrt_g", "gini"),
            ]
        },
    },
}
