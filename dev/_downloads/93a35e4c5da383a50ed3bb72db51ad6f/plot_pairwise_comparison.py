"""
==============================
Ranking by pairwise comparison
==============================

This example shows how :class:`~sklr.reduction.PairwiseLabelRanker`, which fits a
classifier to each pair of labels, turns the probabilities of the pairs into a
ranking, and compares its estimators with
:class:`~sklr.reduction.LabelwiseLabelRanker` on synthetic data, with complete and
incomplete training rankings.
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
# Votes of the labels
# -------------------
#
# The ranker fits a :class:`~sklearn.linear_model.LogisticRegression`, by default,
# to each of the 10 pairs of labels, which predicts the probability that the first
# label of the pair is ahead of the second one. Each label gets as votes the sum of
# the probabilities that it is ahead of each other label, and the predicted ranking
# sorts the labels by decreasing votes. The classifiers are in ``estimators_``, in
# the order of :func:`numpy.triu_indices`, which gives the probabilities and the
# votes of the first sample:

import matplotlib.pyplot as plt

from sklr.reduction import PairwiseLabelRanker

ranker = PairwiseLabelRanker().fit(X, y)
first, second = np.triu_indices(n_labels, k=1)
proba = np.zeros((n_labels, n_labels))
for i, j, estimator in zip(first, second, ranker.estimators_):
    proba[i, j] = estimator.predict_proba(X[:1])[0, 1]
    proba[j, i] = 1 - proba[i, j]
votes = proba.sum(axis=1)
print("Votes:", votes.round(2))
print("Predicted ranking:", ranker.predict(X[:1])[0])
print("True ranking:", y[0])

fig, ax = plt.subplots(figsize=(4.5, 4))
# A label is not compared with itself, so the diagonal is left blank
image = ax.imshow(
    np.where(np.eye(n_labels, dtype=bool), np.nan, proba), vmin=0, vmax=1, cmap="RdBu"
)
ticks = np.arange(n_labels)
ax.set(
    xticks=ticks,
    yticks=ticks,
    xticklabels=[f"$y_{j + 1}$" for j in ticks],
    yticklabels=[f"$y_{i + 1}$" for i in ticks],
    title="Probability that the row label is ahead\n(first sample)",
)
fig.colorbar(image, ax=ax)
fig.tight_layout()

# %%
# Estimators
# ----------
#
# The ranker is scored by 5-fold cross-validation with Kendall's :math:`\tau`, its
# default score, and with Spearman's :math:`\rho`, whose expected value the voting
# maximizes when the probabilities of the classifiers are right, with three
# classifiers: the default logistic regression,
# :class:`~sklearn.svm.LinearSVC`, which has no ``predict_proba`` and is calibrated
# by :class:`~sklearn.calibration.CalibratedClassifierCV`, and
# :class:`~sklearn.tree.DecisionTreeClassifier`. They are compared with
# :class:`~sklr.reduction.LabelwiseLabelRanker` and, as a baseline, with
# :class:`~sklr.dummy.DummyLabelRanker`, which predicts the consensus of the
# training rankings for every sample.

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklr.dummy import DummyLabelRanker
from sklr.metrics import get_scorer
from sklr.reduction import LabelwiseLabelRanker

cv = KFold(n_splits=5, shuffle=True, random_state=0)
scoring = {name: get_scorer(name) for name in ["kendall_tau", "spearman_rho"]}
rankers = {
    "Pairwise,\nLogisticRegression": PairwiseLabelRanker(),
    "Pairwise,\ncalibrated LinearSVC": PairwiseLabelRanker(
        CalibratedClassifierCV(LinearSVC())
    ),
    "Pairwise,\nDecisionTreeClassifier": PairwiseLabelRanker(
        DecisionTreeClassifier(max_depth=4, random_state=0)
    ),
    "LabelwiseLabelRanker": LabelwiseLabelRanker(),
    "DummyLabelRanker": DummyLabelRanker(),
}
scores = {
    name: cross_validate(ranker, X, y, cv=cv, scoring=scoring)
    for name, ranker in rankers.items()
}

fig, ax = plt.subplots(figsize=(8, 4))
width = 0.4
for k, (metric, label) in enumerate(
    [("kendall_tau", r"Kendall's $\tau$"), ("spearman_rho", r"Spearman's $\rho$")]
):
    values = [result[f"test_{metric}"].mean() for result in scores.values()]
    ax.bar(np.arange(len(rankers)) + (k - 0.5) * width, values, width, label=label)
ax.set(
    xticks=np.arange(len(rankers)),
    ylabel="Score (5-fold CV)",
    ylim=(-0.1, 1),
)
ax.set_xticklabels(rankers, fontsize="small")
ax.axhline(0, color="gray", linewidth=0.8)
ax.legend(loc="upper right")
fig.tight_layout()

# %%
# All the rankers that use the features score well above the baseline. The
# utilities are linear in the features, so the linear classifiers score about the
# same, with or without calibration, and about as well as
# :class:`~sklr.reduction.LabelwiseLabelRanker`, while the decision trees, which
# approximate the utilities with thresholds on the features, score lower. Both
# metrics order the rankers in the same way, but they have different scales, so
# each one only compares the rankers, not the metrics. In practice, the estimator
# is chosen by cross-validation, as described in the
# :ref:`user guide <pairwise_comparison>`.
#
# Incomplete training rankings
# ----------------------------
#
# Labels are now removed at random from the training rankings, with a probability
# of 0, 0.3 and 0.6 for each label, and the positions of the remaining ones are
# made dense again. The classifier of each pair is fitted to the training rankings
# that rank both of its labels. The test rankings stay complete, because the
# metrics only compare complete rankings.


def remove_labels(y, probability, random_state):
    """Remove each label of each ranking with the given probability."""
    y = y.copy()
    y[random_state.rand(*y.shape) < probability] = np.nan
    return rankdata(y, axis=1, nan_policy="omit")


probabilities = [0, 0.3, 0.6]
rankers = {
    "PairwiseLabelRanker": PairwiseLabelRanker(),
    "LabelwiseLabelRanker": LabelwiseLabelRanker(),
}
scores = {name: [] for name in rankers}
for probability in probabilities:
    fold_scores = {name: [] for name in rankers}
    for train, test in cv.split(X):
        y_train = remove_labels(y[train], probability, rng)
        for name, ranker in rankers.items():
            ranker.fit(X[train], y_train)
            fold_scores[name].append(ranker.score(X[test], y[test]))
    for name in rankers:
        scores[name].append(np.mean(fold_scores[name]))

fig, ax = plt.subplots(figsize=(5, 4))
for name in rankers:
    ax.plot(probabilities, scores[name], marker="o", label=name)
ax.set(
    xlabel="Probability of removing each training label",
    ylabel=r"Kendall's $\tau$ (5-fold CV)",
    xticks=probabilities,
)
ax.legend()
fig.tight_layout()
plt.show()

# %%
# The scores of both rankers only decrease slightly as more labels are removed.
# With 60% of the labels removed, a ranking ranks both labels of a pair with
# probability :math:`0.4^2 = 0.16`, so the classifier of each pair is fitted to
# about 16% of the training rankings, which are still enough for a linear model of
# three features.
