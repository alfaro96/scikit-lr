"""
================================================
Criteria of a decision tree partial label ranker
================================================

This example compares the criteria of
:class:`~sklr.tree.DecisionTreePartialLabelRanker` on synthetic bucket orders for
several maximum depths, first with complete training rankings and then with labels
removed from them.
"""

# %%
# Synthetic bucket orders
# -----------------------
#
# Each sample has two features and a utility for each of five labels, which is a
# nonlinear function of the features plus some noise. The labels whose utilities
# round to the same value are tied, so the ranking of a sample is a bucket order
# that sorts the buckets by decreasing utility. It holds the dense position of each
# label, as described in :ref:`ranking_representation`.

import numpy as np
from scipy.stats import rankdata

rng = np.random.RandomState(0)
n_samples, n_features, n_labels = 300, 2, 5
X = rng.uniform(-1, 1, size=(n_samples, n_features))
coef = rng.normal(size=(n_features, n_labels))
utilities = np.sin(3 * X) @ coef + 0.3 * rng.normal(size=(n_samples, n_labels))
y = rankdata(-np.round(utilities), method="dense", axis=1)
print(y[:3])

# %%
# Criteria and maximum depth
# --------------------------
#
# Trees of increasing maximum depth are grown with each criterion and scored by
# 5-fold cross-validation with the :math:`\tau_x` coefficient, whose scorer is given
# by :func:`~sklr.metrics.get_scorer`. As a baseline,
# :class:`~sklr.dummy.DummyPartialLabelRanker` predicts the consensus of the
# training rankings for every sample, without looking at the features.

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score

from sklr.dummy import DummyPartialLabelRanker
from sklr.metrics import get_scorer
from sklr.tree import DecisionTreePartialLabelRanker

cv = KFold(n_splits=5, shuffle=True, random_state=0)
scoring = get_scorer("tau_x")
max_depths = np.arange(1, 11)
criteria = ["disagreements", "distance", "entropy", "gini"]

fig, ax = plt.subplots(figsize=(5, 4))
for criterion in criteria:
    scores = [
        cross_val_score(
            DecisionTreePartialLabelRanker(
                criterion=criterion, max_depth=max_depth, random_state=0
            ),
            X,
            y,
            scoring=scoring,
            cv=cv,
        ).mean()
        for max_depth in max_depths
    ]
    ax.plot(max_depths, scores, marker="o", label=f"criterion={criterion!r}")
baseline = cross_val_score(
    DummyPartialLabelRanker(), X, y, scoring=scoring, cv=cv
).mean()
ax.axhline(baseline, color="gray", linestyle="--", label="DummyPartialLabelRanker")
ax.set(xlabel="max_depth", ylabel=r"$\tau_x$ (5-fold CV)")
ax.legend()
fig.tight_layout()

# %%
# Every tree scores above the baseline. The score grows with the depth while the splits
# separate bucket orders that differ, and stops improving once the leaves fit the noise
# of few rankings. The criteria choose different splits, and on these data the ones that
# compare the rankings with the consensus of each child, ``"disagreements"`` and
# ``"distance"``, reach a slightly higher score. They are also slower here, as they find
# that consensus for each candidate split, as explained in the
# :ref:`user guide <partial_label_ranking_tree>`. As the other parameters that limit the
# size of the tree, the criterion is chosen by cross-validation.
#
# Incomplete training rankings
# ----------------------------
#
# Labels are now removed at random from the training rankings, with a probability
# of 0.3 and 0.6 for each label, and the positions of the remaining ones are made
# dense again. The same labels are removed for all the depths. The validation
# rankings stay complete, because the metrics only compare complete rankings. The
# trees use the default criterion, ``"entropy"``.


def remove_labels(y, probability, random_state):
    """Remove each label of each ranking with the given probability."""
    y = y.copy()
    y[random_state.rand(*y.shape) < probability] = np.nan
    return rankdata(y, method="dense", axis=1, nan_policy="omit")


fig, ax = plt.subplots(figsize=(5, 4))
for probability in [0, 0.3, 0.6]:
    folds = [
        (train, test, remove_labels(y[train], probability, rng))
        for train, test in cv.split(X)
    ]
    scores = []
    for max_depth in max_depths:
        ranker = DecisionTreePartialLabelRanker(max_depth=max_depth, random_state=0)
        fold_scores = [
            ranker.fit(X[train], y_train).score(X[test], y[test])
            for train, test, y_train in folds
        ]
        scores.append(np.mean(fold_scores))
    ax.plot(max_depths, scores, marker="o", label=f"Probability {probability}")
ax.set(xlabel="max_depth", ylabel=r"$\tau_x$ (5-fold CV)")
ax.legend(title="Removed labels")
fig.tight_layout()
plt.show()

# %%
# The tree learns from the incomplete rankings as they are: each pair of labels
# counts only in the rankings that rank both, and the labels that no ranking of a
# leaf ranks together with another one are tied in a last bucket. With fewer ranked
# labels, each leaf holds less information about the bucket order, so the deep
# trees lose more and the best depth is smaller.
