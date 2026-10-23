"""Protocol of [cheng_labelwise_2013] for LWD."""

from experiments.datasets import missing_labels_scenarios
from sklearn.model_selection import RepeatedKFold

from sklr.reduction import LabelwiseLabelRanker
from sklr.reduction._ordinal import OrdinalLogisticRegression

# The paper does not say how it fits its ordinal logistic regression, so it has the
# ridge of the default logistic regression of Weka, which adds the ridge times the
# squared norm of the coefficients, but not of the thresholds, to the negative
# log-likelihood, half the alpha of the ordinal logistic regression. It is fitted
# until it converges
WEKA_RIDGE = 1e-8

PAPER = {
    # The semi-synthetic datasets of Table 1 (p. 134). Its real-world datasets,
    # sushi and students, could not be obtained
    "datasets": {
        "tag": "label-ranking",
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
    "metrics": ["kendall_tau"],
    # Five repetitions of a ten-fold cross-validation (p. 134)
    "cross_validation": {
        "outer": RepeatedKFold(n_splits=10, n_repeats=5, random_state=0),
    },
    # The paper tunes no parameter, and does not say which loss its rankings
    # minimize, so it is the default one
    "methods": {
        "lwd": {
            "estimator": LabelwiseLabelRanker,
            "params": {
                "estimator": OrdinalLogisticRegression(
                    alpha=2 * WEKA_RIDGE, max_iter=10_000
                )
            },
        },
    },
}
