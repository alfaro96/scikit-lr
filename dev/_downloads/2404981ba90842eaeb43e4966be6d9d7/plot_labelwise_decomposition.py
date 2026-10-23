"""
===================================
Labelwise decomposition of rankings
===================================

This example compares the losses and the estimators of
:class:`~sklr.reduction.LabelwiseLabelRanker`, which fits a classifier to the
positions of each label, on synthetic data. It first scores them with complete
rankings, and then removes labels from the training rankings to see how the default
estimator learns from incomplete rankings.
"""

# %%
# Synthetic rankings
# ------------------
#
# Each sample has three features and a utility for each of five labels, which is a
# linear function of the features plus some noise. The ranking of a sample sorts
# the labels by decreasing utility, and holds the position of each label, as
# described in :ref:`ranking_representation`.

import numpy as np
from scipy.stats import rankdata

rng = np.random.RandomState(0)
n_samples, n_features, n_labels = 300, 3, 5
X = rng.uniform(-1, 1, size=(n_samples, n_features))
coef = rng.normal(size=(n_features, n_labels))
utilities = X @ coef + 0.5 * rng.normal(size=(n_samples, n_labels))
y = rankdata(-utilities, axis=1)
print(y[:3])

# %%
# Losses and estimators
# ---------------------
#
# Each combination of loss and estimator is scored by 5-fold cross-validation with
# Kendall's :math:`\tau`, the default score of the label rankers. The default
# estimator is an ordinal logistic regression, which takes the positions as ordered
# classes. :class:`~sklearn.linear_model.LogisticRegression` and
# :class:`~sklearn.tree.DecisionTreeClassifier` take them as unordered classes. As a
# baseline, :class:`~sklr.dummy.DummyLabelRanker` predicts the consensus of the
# training rankings for every sample, without looking at the features.

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklr.dummy import DummyLabelRanker
from sklr.reduction import LabelwiseLabelRanker

cv = KFold(n_splits=5, shuffle=True, random_state=0)
estimators = {
    "Ordinal logistic regression (default)": None,
    "LogisticRegression": LogisticRegression(),
    "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=4, random_state=0),
}
losses = ["absolute_error", "squared_error"]

scores = {
    name: [
        cross_val_score(LabelwiseLabelRanker(estimator, loss=loss), X, y, cv=cv).mean()
        for loss in losses
    ]
    for name, estimator in estimators.items()
}
baseline = cross_val_score(DummyLabelRanker(), X, y, cv=cv).mean()

fig, ax = plt.subplots(figsize=(7, 4))
width = 0.25
for i, (name, values) in enumerate(scores.items()):
    ax.bar(np.arange(len(losses)) + (i - 1) * width, values, width, label=name)
ax.axhline(baseline, color="gray", linestyle="--", label="DummyLabelRanker")
ax.set(
    xticks=np.arange(len(losses)),
    xticklabels=[f"loss={loss!r}" for loss in losses],
    ylabel=r"Kendall's $\tau$ (5-fold CV)",
    ylim=(0, 1),
)
ax.legend(loc="lower right")
fig.tight_layout()

# %%
# All the rankers score well above the baseline, so the features carry information
# about the rankings. The two losses score about the same, although they may
# predict different rankings for some samples. Both logistic regressions are
# linear in the features, like the utilities, and score about the same, while the
# decision trees, which approximate the linear utilities with thresholds on the
# features, score lower. In practice, the estimator and the loss are chosen by
# cross-validation, as described in the :ref:`user guide <labelwise_decomposition>`.
#
# Incomplete training rankings
# ----------------------------
#
# Labels are now removed at random from the training rankings, with a probability
# of 0, 0.3 and 0.6 for each label, and the positions of the remaining ones are
# made dense again. Of the estimators above, only the default one learns from
# incomplete rankings, since it is the only one with a ``fit_likelihood`` method,
# and the ranker assumes that their labels were removed at random. Both losses are
# fitted to the same incomplete rankings, and the test rankings stay complete,
# because the metrics only compare complete rankings.


def remove_labels(y, probability, random_state):
    """Remove each label of each ranking with the given probability."""
    y = y.copy()
    y[random_state.rand(*y.shape) < probability] = np.nan
    return rankdata(y, axis=1, nan_policy="omit")


probabilities = [0, 0.3, 0.6]
scores = {loss: [] for loss in losses}
for probability in probabilities:
    fold_scores = {loss: [] for loss in losses}
    for train, test in cv.split(X):
        y_train = remove_labels(y[train], probability, rng)
        for loss in losses:
            ranker = LabelwiseLabelRanker(loss=loss).fit(X[train], y_train)
            fold_scores[loss].append(ranker.score(X[test], y[test]))
    for loss in losses:
        scores[loss].append(np.mean(fold_scores[loss]))

fig, ax = plt.subplots(figsize=(5, 4))
for loss in losses:
    ax.plot(probabilities, scores[loss], marker="o", label=f"loss={loss!r}")
ax.set(
    xlabel="Probability of removing each training label",
    ylabel=r"Kendall's $\tau$ (5-fold CV)",
    xticks=probabilities,
)
ax.legend()
fig.tight_layout()
plt.show()

# %%
# The scores only decrease slightly as more labels are removed, even when more than
# half of them are missing, because the likelihood of the classifier of each label
# takes into account every position that the label may have in the complete
# ranking, with its probability given that the other labels were removed at random.
