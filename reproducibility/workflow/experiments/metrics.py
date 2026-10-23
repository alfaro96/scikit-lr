"""Metrics of the papers and scorers of incomplete validation rankings."""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import make_scorer

from sklr.metrics import kendall_tau_score, spearman_rho_score, tau_x_score


def observed_score(y_true, y_pred, metric):
    """Score the predictions on the labels ranked by each true ranking.

    The inner cross-validation chooses the parameters of a method from the training
    rankings of an outer fold, which are incomplete in the scenarios that remove
    labels, so its validation rankings are incomplete too. The metrics of
    scikit-lr only compare complete rankings, and the papers do not say how they
    score a prediction against an incomplete ranking. The only information that
    such a ranking gives is how it orders the labels that it ranks, so each
    prediction is scored with `metric` on those labels alone, after making the
    positions of the predicted ranking dense among them. Each pair of labels that
    the true ranking ranks then counts as with complete rankings, and the pairs
    with a label that it does not rank do not count. The rankings with fewer than
    two ranked labels give no pair to compare, so they are left out, and the score
    is the mean over the others, or ``0`` if there are none.

    The rows are grouped by the labels that their true ranking ranks, so that the
    metric is called once per group instead of once per row.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ranked = ~np.isnan(y_true)
    masks, groups = np.unique(ranked, axis=0, return_inverse=True)
    total, n_scored = 0.0, 0
    for group, mask in enumerate(masks):
        if mask.sum() < 2:
            continue
        rows = groups == group
        true = y_true[np.ix_(rows, mask)]
        pred = rankdata(y_pred[np.ix_(rows, mask)], method="dense", axis=1)
        total += metric(true, pred) * rows.sum()
        n_scored += rows.sum()
    return total / n_scored if n_scored else 0.0


# Each metric of the papers, with its name in LaTeX for the report and the figures,
# the function that scores the test rankings and the scorer of the inner
# cross-validation, whose validation rankings may be incomplete
METRICS = {
    "kendall_tau": {
        "latex": r"\text{Kendall's }\tau",
        "score": kendall_tau_score,
        "inner_scorer": make_scorer(observed_score, metric=kendall_tau_score),
    },
    "spearman_rho": {
        "latex": r"\text{Spearman's }\rho",
        "score": spearman_rho_score,
        "inner_scorer": make_scorer(observed_score, metric=spearman_rho_score),
    },
    "tau_x": {
        "latex": r"\tau_x",
        "score": tau_x_score,
        "inner_scorer": make_scorer(observed_score, metric=tau_x_score),
    },
}
