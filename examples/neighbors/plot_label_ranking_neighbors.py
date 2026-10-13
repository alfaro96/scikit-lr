"""
==========================================
Label ranking with the k-nearest neighbors
==========================================

This example compares the two probability models of
:class:`~sklr.neighbors.KNeighborsLabelRanker`, the Mallows and the Plackett-Luce
model, and its weights of the neighbors on synthetic data. It first scores them with
complete rankings for several numbers of neighbors, and then removes labels from the
training rankings to see how each model learns from incomplete rankings.
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
n_samples, n_features, n_labels = 300, 2, 5
X = rng.uniform(-1, 1, size=(n_samples, n_features))
coef = rng.normal(size=(n_features, n_labels))
utilities = np.sin(3 * X) @ coef + 0.3 * rng.normal(size=(n_samples, n_labels))
y = rankdata(-utilities, axis=1)
print(y[:3])

# %%
# Number of neighbors and weights
# -------------------------------
#
# Each combination of model and weights is scored by 5-fold cross-validation with
# Kendall's :math:`\tau`, whose scorer is given by :func:`~sklr.metrics.get_scorer`.
# Both features have the same range, so they are not scaled.
#
# The maximum likelihood estimate of the Plackett-Luce model does not exist when
# some label is never ranked above another one in a neighborhood, which is common
# with few neighbors. Its estimation then issues a
# :class:`~sklearn.exceptions.ConvergenceWarning`, and the prediction sorts the
# labels by the parameters reached after ``max_iter`` iterations. These warnings are
# expected here, so they are silenced.

import warnings

import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, cross_val_score

from sklr.metrics import get_scorer
from sklr.neighbors import KNeighborsLabelRanker

cv = KFold(n_splits=5, shuffle=True, random_state=0)
scoring = get_scorer("kendall_tau")
n_neighbors_grid = [1, 5, 10, 20, 40]
models = ["mallows", "plackett_luce"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, model in zip(axes, models):
    for weights in ["uniform", "distance", "dudani"]:
        scores = []
        for n_neighbors in n_neighbors_grid:
            ranker = KNeighborsLabelRanker(n_neighbors, model=model, weights=weights)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                score = cross_val_score(ranker, X, y, scoring=scoring, cv=cv).mean()
            scores.append(score)
        ax.plot(n_neighbors_grid, scores, marker="o", label=f"weights={weights!r}")
    ax.set(title=f"model={model!r}", xlabel="Number of neighbors")
    ax.legend()
axes[0].set_ylabel(r"Kendall's $\tau$ (5-fold CV)")
fig.tight_layout()

# %%
# With a single neighbor, both models predict its ranking, whatever the weights.
# More neighbors average out the noise of the rankings, until they come from too
# far away. Weighting the neighbors by the inverse of their distance or by the rule
# of Dudani lets the ranker use more neighbors before the far ones degrade the
# prediction. In practice, the number of neighbors and the weights are chosen by
# cross-validation, as described in the :ref:`user guide <neighbors>`.
#
# Incomplete training rankings
# ----------------------------
#
# Labels are now removed at random from the training rankings, with a probability
# of 0, 0.3 and 0.6 for each label, and the positions of the remaining ones are
# made dense again. The test rankings stay complete, because the metrics only
# compare complete rankings. Both models weigh the neighbors uniformly, so that only
# the model differs.


def remove_labels(y, probability, random_state):
    """Remove each label of each ranking with the given probability."""
    y = y.copy()
    y[random_state.rand(*y.shape) < probability] = np.nan
    return rankdata(y, axis=1, nan_policy="omit")


probabilities = [0, 0.3, 0.6]
scores = {model: [] for model in models}
for probability in probabilities:
    for model in models:
        fold_scores = []
        for train, test in cv.split(X):
            y_train = remove_labels(y[train], probability, rng)
            ranker = KNeighborsLabelRanker(10, model=model).fit(X[train], y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fold_scores.append(ranker.score(X[test], y[test]))
        scores[model].append(np.mean(fold_scores))

fig, ax = plt.subplots(figsize=(5, 4))
for model in models:
    ax.plot(probabilities, scores[model], marker="o", label=f"model={model!r}")
ax.set(
    xlabel="Probability of removing each training label",
    ylabel=r"Kendall's $\tau$ (5-fold CV)",
    xticks=probabilities,
)
ax.legend()
fig.tight_layout()
plt.show()

# %%
# Both models score about the same with complete rankings. As more labels are
# removed, the Mallows model, which completes the incomplete rankings of the
# neighbors with its own center, loses more than the Plackett-Luce model, whose
# likelihood uses the incomplete rankings as they are.
