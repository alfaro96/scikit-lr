"""Protocol of [hullermeier_label_2008] for RPC."""

from experiments.datasets import missing_labels_scenarios
from sklearn.model_selection import RepeatedKFold

from sklr.reduction import PairwiseLabelRanker

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
    # The paper tunes no parameter, and its logistic regression has no stated
    # settings, so it is the default one of scikit-learn
    "methods": {
        "rpc": {"estimator": PairwiseLabelRanker, "params": {}},
    },
}
