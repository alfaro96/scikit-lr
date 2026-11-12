"""
======================================
Choosing a label ranker by grid search
======================================

This example chooses between :class:`~sklr.neighbors.KNeighborsLabelRanker` and
:class:`~sklr.tree.DecisionTreeLabelRanker`, and their parameters, with a
:class:`~sklearn.pipeline.Pipeline` and :class:`~sklearn.model_selection.GridSearchCV`,
and estimates the score of the chosen ranker with nested cross-validation.
"""

# %%
# Synthetic rankings
# ------------------
#
# Each sample has three features and a utility for each of five labels, which is a
# nonlinear function of the features plus some noise. The third feature does not
# change the utilities, and its scale is much larger than that of the other two.
# The ranking of a sample sorts the labels by decreasing utility, and holds the
# position of each label, as described in :ref:`ranking_representation`.

import numpy as np
from scipy.stats import rankdata

rng = np.random.RandomState(0)
n_samples, n_labels = 300, 5
X = np.column_stack(
    [rng.uniform(-1, 1, size=(n_samples, 2)), rng.normal(scale=100, size=n_samples)]
)
coef = rng.normal(size=(2, n_labels))
utilities = np.sin(3 * X[:, :2]) @ coef + 0.3 * rng.normal(size=(n_samples, n_labels))
y = rankdata(-utilities, axis=1)
print(y[:3])

# %%
# Grid search
# -----------
#
# The pipeline scales the features and then fits a ranker. The grid is a list of two
# grids, one per ranker, whose parameters are prefixed with the name of their step.
# The nearest neighbors ranker is tried with and without the scaler, which the
# string ``"passthrough"`` skips, and the tree only without it, since its splits do
# not depend on the scale of the features. The candidates are scored by 5-fold
# cross-validation with Kendall's :math:`\tau`, whose scorer is given by
# :func:`~sklr.metrics.get_scorer`, and fitted in parallel with ``n_jobs``.
#
# The tree has a fixed ``random_state``, so that the search gives the same result
# with any number of jobs.

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklr.metrics import get_scorer
from sklr.neighbors import KNeighborsLabelRanker
from sklr.tree import DecisionTreeLabelRanker

pipeline = Pipeline([("scaler", StandardScaler()), ("ranker", None)])
n_neighbors = [1, 3, 5, 10, 20, 40]
max_depth = [1, 2, 3, 4, 6, 8]
param_grid = [
    {
        "scaler": [StandardScaler(), "passthrough"],
        "ranker": [KNeighborsLabelRanker()],
        "ranker__n_neighbors": n_neighbors,
    },
    {
        "scaler": ["passthrough"],
        "ranker": [DecisionTreeLabelRanker(random_state=0)],
        "ranker__max_depth": max_depth,
    },
]
inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
search = GridSearchCV(
    pipeline, param_grid, scoring=get_scorer("kendall_tau"), cv=inner_cv, n_jobs=2
)
search.fit(X, y)
print(search.best_params_)

# %%
# The mean score of each candidate is in ``cv_results_``. The masks select the
# candidates of each ranker and scaler, whose scores are plotted against the
# parameter that the grid changes for them.

import matplotlib.pyplot as plt

results = search.cv_results_
is_neighbors = np.array(
    [isinstance(ranker, KNeighborsLabelRanker) for ranker in results["param_ranker"]]
)
is_scaled = np.array(
    [isinstance(scaler, StandardScaler) for scaler in results["param_scaler"]]
)
scores = results["mean_test_score"]

fig, (ax_neighbors, ax_tree) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for mask, label in [
    (is_neighbors & is_scaled, "Scaled features"),
    (is_neighbors & ~is_scaled, "Original features"),
]:
    ax_neighbors.plot(n_neighbors, scores[mask], marker="o", label=label)
ax_neighbors.set(
    title="KNeighborsLabelRanker",
    xlabel="n_neighbors",
    ylabel=r"Kendall's $\tau$ (5-fold CV)",
)
ax_neighbors.legend()
ax_tree.plot(max_depth, scores[~is_neighbors], marker="o")
ax_tree.set(title="DecisionTreeLabelRanker", xlabel="max_depth")
fig.tight_layout()

# %%
# Without the scaler, the distances between the samples depend almost only on the
# third feature, which has the largest scale and says nothing about the rankings,
# so the neighbors are no better than random samples. Once the features are scaled,
# the two informative ones count again, but the third one still adds noise to the
# distances. The tree is not affected by the scale and splits on the informative
# features, so it gets the best score, which grows with the depth until its leaves
# fit the noise of few rankings.
#
# Nested cross-validation
# -----------------------
#
# The best score of the search is an optimistic estimate of the score of the chosen
# ranker, because the same folds chose it among many candidates. Nested
# cross-validation estimates it without this bias: the whole search is the
# estimator of an outer cross-validation, so each outer fold chooses a ranker
# without its test samples and then scores it on them. The best pipeline of each
# outer fold is in the ``best_estimator_`` of its search. The outer folds are fitted
# in parallel, so the inner searches run with a single job.

from sklearn.model_selection import cross_validate

outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
nested = cross_validate(
    search.set_params(n_jobs=1), X, y, cv=outer_cv, n_jobs=2, return_estimator=True
)
print(f"Best score of the search: {search.best_score_:.3f}")
print(f"Nested cross-validation:  {nested['test_score'].mean():.3f}")
for fitted in nested["estimator"]:
    print(fitted.best_estimator_["ranker"])

# %%
# The nested estimate is lower than the best score of the search. Every outer fold
# chooses the tree, with a maximum depth of 3 or 4, so the choice is stable.

plt.show()
