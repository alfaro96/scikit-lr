"""Protocol of [cheng_labelwise_2013] for LWD."""

from experiments.datasets import missing_labels_scenarios
from sklearn.model_selection import RepeatedKFold

from sklr.reduction import LabelwiseLabelRanker

PAPER = {
    # The semi-synthetic datasets of Table 1 (p. 134). Its real-world datasets,
    # sushi and students, could not be obtained
    "datasets": {
        "tag": "label ranking",
        "names": [
            "authorship",
            "glass",
            "iris",
            "pendigits",
            "segment",
            "vehicle",
            "vowel",
            "wine",
        ],
    },
    # The labels are removed at random, the scenario of the upper half of Table 2
    # (p. 135)
    "scenarios": missing_labels_scenarios([0, 0.3, 0.6]),
    "metric": "kendall_tau",
    # Five repetitions of a ten-fold cross-validation (p. 134)
    "cross_validation": {
        "outer": RepeatedKFold(n_splits=10, n_repeats=5, random_state=0),
    },
    # The paper tunes no parameter, and does not say which loss its rankings
    # minimize, so it is the default one
    "methods": {
        "lwd": {"estimator": LabelwiseLabelRanker, "params": {}},
    },
}
