"""
==================================================
Partial label ranking with the k-nearest neighbors
==================================================

This example compares the weights of the neighbors of
:class:`~sklr.neighbors.KNeighborsPartialLabelRanker` on synthetic bucket orders. It
first scores them for several numbers of neighbors, and then removes labels from the
training rankings to see how the consensus of the neighbors copes with incomplete
partial rankings.
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
# Number of neighbors and weights
# -------------------------------
#
# Each weight function is scored by 5-fold cross-validation with the :math:`\tau_x`
# coefficient, whose scorer is given by :func:`~sklr.metrics.get_scorer`. Both
# features have the same range, so they are not scaled. As a baseline,
# :class:`~sklr.dummy.DummyPartialLabelRanker` predicts the consensus of the
# training rankings for every sample, without looking at the features.

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score

from sklr.dummy import DummyPartialLabelRanker
from sklr.metrics import get_scorer
from sklr.neighbors import KNeighborsPartialLabelRanker

cv = KFold(n_splits=5, shuffle=True, random_state=0)
scoring = get_scorer("tau_x")
n_neighbors_grid = [1, 5, 10, 20, 40]
weights_grid = ["uniform", "distance", "dudani"]

fig, ax = plt.subplots(figsize=(5, 4))
for weights in weights_grid:
    scores = []
    for n_neighbors in n_neighbors_grid:
        ranker = KNeighborsPartialLabelRanker(n_neighbors, weights=weights)
        scores.append(cross_val_score(ranker, X, y, scoring=scoring, cv=cv).mean())
    ax.plot(n_neighbors_grid, scores, marker="o", label=f"weights={weights!r}")
baseline = cross_val_score(
    DummyPartialLabelRanker(), X, y, scoring=scoring, cv=cv
).mean()
ax.axhline(baseline, color="gray", linestyle="--", label="DummyPartialLabelRanker")
ax.set(xlabel="Number of neighbors", ylabel=r"$\tau_x$ (5-fold CV)")
ax.legend()
fig.tight_layout()

# %%
# All the rankers score well above the baseline, so the features carry information about
# the bucket orders. With a single neighbor, the ranker predicts its bucket order,
# whatever the weights. More neighbors average out the noise of the rankings, until they
# come from too far away. Weighting the neighbors by the inverse of their distance or by
# the rule of Dudani lets the ranker use more neighbors before the far ones degrade the
# prediction. In practice, the number of neighbors and the weights are chosen by
# cross-validation, as described in the
# :ref:`user guide <partial_label_ranking_neighbors>`.
#
# Incomplete training rankings
# ----------------------------
#
# Labels are now removed at random from the training rankings, with a probability
# of 0, 0.3 and 0.6 for each label, and the positions of the remaining ones are
# made dense again. The test rankings stay complete, because :math:`\tau_x` only
# compares complete rankings. The neighbors are weighted by the inverse of their
# distance, and several numbers of neighbors are compared.


def remove_labels(y, probability, random_state):
    """Remove each label of each ranking with the given probability."""
    y = y.copy()
    y[random_state.rand(*y.shape) < probability] = np.nan
    return rankdata(y, method="dense", axis=1, nan_policy="omit")


probabilities = [0, 0.3, 0.6]
n_neighbors_grid = [5, 10, 20]
scores = {n_neighbors: [] for n_neighbors in n_neighbors_grid}
for probability in probabilities:
    for n_neighbors in n_neighbors_grid:
        fold_scores = []
        for train, test in cv.split(X):
            y_train = remove_labels(y[train], probability, rng)
            ranker = KNeighborsPartialLabelRanker(n_neighbors, weights="distance")
            ranker.fit(X[train], y_train)
            fold_scores.append(ranker.score(X[test], y[test]))
        scores[n_neighbors].append(np.mean(fold_scores))

fig, ax = plt.subplots(figsize=(5, 4))
for n_neighbors in n_neighbors_grid:
    ax.plot(
        probabilities, scores[n_neighbors], marker="o", label=f"{n_neighbors} neighbors"
    )
ax.set(
    xlabel="Probability of removing each training label",
    ylabel=r"$\tau_x$ (5-fold CV)",
    xticks=probabilities,
)
ax.legend()
fig.tight_layout()
plt.show()

# %%
# The consensus only uses, for each pair of labels, the neighbors that rank both,
# so removing labels leaves fewer rankings to decide each pair. The score drops as
# more labels are removed, and more neighbors make up for part of the loss, since
# each pair is then still ranked by enough of them.
