"""
=====================================
Depth of a decision tree label ranker
=====================================

This example shows how the maximum depth of
:class:`~sklr.tree.DecisionTreeLabelRanker` controls the fit to the training
rankings on synthetic data, first with complete training rankings and then with
labels removed from them.
"""

# %%
# Synthetic rankings
# ------------------
#
# Each sample has two features and a utility for each of five labels, which is a
# nonlinear function of the features plus some noise. The ranking of a sample sorts
# the labels by decreasing utility, and holds the position of each label, as
# described in :ref:`ranking_representation`.

import numpy as np
from scipy.stats import rankdata

rng = np.random.RandomState(0)
n_samples, n_features, n_labels = 200, 2, 5
X = rng.uniform(-1, 1, size=(n_samples, n_features))
coef = rng.normal(size=(n_features, n_labels))
utilities = np.sin(3 * X) @ coef + 0.3 * rng.normal(size=(n_samples, n_labels))
y = rankdata(-utilities, axis=1)
print(y[:3])

# %%
# Maximum depth
# -------------
#
# Trees of increasing maximum depth are scored by 5-fold cross-validation with
# Kendall's :math:`\tau`, whose scorer is given by :func:`~sklr.metrics.get_scorer`,
# both on the training and on the validation folds. As a baseline,
# :class:`~sklr.dummy.DummyLabelRanker` predicts the consensus of the training
# rankings for every sample, without looking at the features.

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, cross_validate

from sklr.dummy import DummyLabelRanker
from sklr.metrics import get_scorer
from sklr.tree import DecisionTreeLabelRanker

cv = KFold(n_splits=5, shuffle=True, random_state=0)
scoring = get_scorer("kendall_tau")
max_depths = np.arange(1, 11)

train_scores, test_scores = [], []
for max_depth in max_depths:
    ranker = DecisionTreeLabelRanker(max_depth=max_depth, random_state=0)
    results = cross_validate(
        ranker, X, y, scoring=scoring, cv=cv, return_train_score=True
    )
    train_scores.append(results["train_score"].mean())
    test_scores.append(results["test_score"].mean())

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(max_depths, train_scores, marker="o", label="Training folds")
ax.plot(max_depths, test_scores, marker="o", label="Validation folds")
baseline = cross_val_score(DummyLabelRanker(), X, y, scoring=scoring, cv=cv).mean()
ax.axhline(baseline, color="gray", linestyle="--", label="DummyLabelRanker")
ax.set(xlabel="max_depth", ylabel=r"Kendall's $\tau$ (5-fold CV)")
ax.legend()
fig.tight_layout()

# %%
# Even a tree with a single split scores well above the baseline, so the features carry
# information about the rankings. The score on the training folds keeps growing with the
# depth, because each new level separates the training rankings further. The score on
# the validation folds stops improving after a few levels, and the gap between both
# grows as the leaves fit the noise of fewer and fewer rankings. In practice, the depth,
# or the other parameters that limit the size of the tree, are chosen by
# cross-validation, as described in the :ref:`user guide <tree>`.
#
# Incomplete training rankings
# ----------------------------
#
# Labels are now removed at random from the training rankings, with a probability
# of 0.3 and 0.6 for each label, and the positions of the remaining ones are made
# dense again. The same labels are removed for all the depths. The validation
# rankings stay complete, because the metrics only compare complete rankings, and
# the scores with complete training rankings are those of the validation folds
# above.


def remove_labels(y, probability, random_state):
    """Remove each label of each ranking with the given probability."""
    y = y.copy()
    y[random_state.rand(*y.shape) < probability] = np.nan
    return rankdata(y, axis=1, nan_policy="omit")


fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(max_depths, test_scores, marker="o", label="Probability 0")
for probability in [0.3, 0.6]:
    folds = [
        (train, test, remove_labels(y[train], probability, rng))
        for train, test in cv.split(X)
    ]
    scores = []
    for max_depth in max_depths:
        ranker = DecisionTreeLabelRanker(max_depth=max_depth, random_state=0)
        fold_scores = [
            ranker.fit(X[train], y_train).score(X[test], y[test])
            for train, test, y_train in folds
        ]
        scores.append(np.mean(fold_scores))
    ax.plot(max_depths, scores, marker="o", label=f"Probability {probability}")
ax.set(xlabel="max_depth", ylabel=r"Kendall's $\tau$ (5-fold CV)")
ax.legend(title="Removed labels")
fig.tight_layout()
plt.show()

# %%
# The tree learns from the incomplete rankings as they are: the model of each node
# is estimated on the labels that its rankings rank, and the labels that no ranking
# of a leaf ranks take their place from its ancestors. With fewer ranked labels,
# each leaf holds less information about the ranking, so the deep trees lose more
# and the best depth is smaller.
