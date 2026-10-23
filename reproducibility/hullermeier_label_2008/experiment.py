"""Protocol of [hullermeier_label_2008] for RPC."""

from experiments.datasets import missing_labels_scenarios
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklr.reduction import PairwiseLabelRanker

# The ridge of the default logistic regression of Weka, which the paper uses (p. 1908,
# footnote 10). Weka adds the ridge times the squared norm of the coefficients, but
# not of the intercept, to the negative log-likelihood of the standardized features,
# so it is half the alpha of LogisticRegression, or the inverse of twice its C
WEKA_RIDGE = 1e-8


def weka_logistic_regression():
    """Build the default logistic regression of Weka, fitted until it converges.

    The older releases of scikit-learn have no alpha, but its inverse, C, and the
    newer ones warn when C is set.
    """
    if "alpha" in LogisticRegression().get_params():
        regression = LogisticRegression(alpha=2 * WEKA_RIDGE, max_iter=10_000)
    else:
        regression = LogisticRegression(C=1 / (2 * WEKA_RIDGE), max_iter=10_000)
    return make_pipeline(StandardScaler(), regression)


PAPER = {
    # The semi-synthetic and the yeast datasets of Table 2 (p. 1908)
    "datasets": {
        "tag": "label-ranking",
        "names": [
            "cold",
            "diau",
            "dtt",
            "glass",
            "heat",
            "iris",
            "spo",
            "vehicle",
            "vowel",
            "wine",
        ],
    },
    # Complete training rankings (Tables 3 and 4, p. 1909), and from 10% to 70% of
    # their labels removed at random (Fig. 4, p. 1911)
    "scenarios": missing_labels_scenarios([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    "metrics": ["kendall_tau", "spearman_rho"],
    # Five repetitions of a ten-fold cross-validation (p. 1909)
    "cross_validation": {
        "outer": RepeatedKFold(n_splits=10, n_repeats=5, random_state=0),
    },
    # The paper tunes no parameter
    "methods": {
        "rpc": {
            "estimator": PairwiseLabelRanker,
            "params": {"estimator": weka_logistic_regression()},
        },
    },
}
