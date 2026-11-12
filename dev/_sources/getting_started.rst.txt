.. _getting_started:

Getting started
===============

The estimators of scikit-lr follow the scikit-learn API, so the scikit-learn
:doc:`getting started guide <sklearn:getting_started>` also applies to them. This
guide shows how they are fitted, how they predict and how they are evaluated and
tuned with the tools of scikit-learn. The :ref:`user guide <user_guide>` gives more
details on each of them.

Fitting and predicting
----------------------

A label ranker learns to predict a ranking of a fixed set of labels for each
sample. Here is a simple example where we fit a
:class:`~sklr.neighbors.KNeighborsLabelRanker` to four samples and three labels:

>>> from sklr.neighbors import KNeighborsLabelRanker
>>> ranker = KNeighborsLabelRanker(n_neighbors=3)
>>> X = [[0], [1], [2], [3]]  # 4 samples, 1 feature
>>> y = [[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]]  # rankings of 3 labels
>>> ranker.fit(X, y)
KNeighborsLabelRanker(n_neighbors=3)

The ``fit`` method accepts the samples matrix ``X``, of shape
``(n_samples, n_features)``, and the rankings ``y``, of shape
``(n_samples, n_labels)``. Each row of ``y`` holds the position of each label in
the ranking of a sample, from ``1`` for the most preferred one, so the second
sample prefers the second label to the first one and the first one to the third
one. The training rankings may be incomplete, with ``np.nan`` for the labels that
they do not rank, as described in :ref:`ranking_representation`.

Once the ranker is fitted, it predicts the rankings of new samples:

>>> ranker.predict([[1.1], [2.2]])
array([[1, 2, 3],
       [2, 1, 3]])

Evaluation
----------

The ``score`` method of a label ranker computes the mean Kendall's
:math:`\tau` between the true and the predicted rankings, from ``-1`` when they are
reversed to ``1`` when they are equal:

>>> ranker.score([[1.1], [2.2]], [[1, 2, 3], [1, 2, 3]])
0.666...

The model selection tools of scikit-learn, such as
:func:`~sklearn.model_selection.cross_val_score`, use the ``score`` method by
default, and the scorers of scikit-lr when they are passed as ``scoring``, which
:func:`~sklr.metrics.get_scorer` gets from their names:

>>> import numpy as np
>>> from scipy.stats import rankdata
>>> from sklearn.model_selection import cross_val_score
>>> from sklr.metrics import get_scorer
>>> rng = np.random.RandomState(0)
>>> X = rng.normal(size=(60, 2))
>>> y = rankdata(-X @ rng.normal(size=(2, 4)), axis=1)  # rankings of 4 labels
>>> scores = cross_val_score(ranker, X, y, scoring=get_scorer("kendall_tau"))
>>> scores.round(2)
array([0.94, 0.86, 0.86, 0.94, 0.94])

:ref:`model_evaluation` describes the metrics and the scorers.

Pipelines and parameter search
------------------------------

A label ranker can be the last step of a :class:`~sklearn.pipeline.Pipeline`,
which here scales the features before searching the neighbors, and its
parameters can be tuned with :class:`~sklearn.model_selection.GridSearchCV`:

>>> from sklearn.model_selection import GridSearchCV
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> pipeline = make_pipeline(StandardScaler(), KNeighborsLabelRanker())
>>> param_grid = {"kneighborslabelranker__n_neighbors": [1, 5, 10]}
>>> search = GridSearchCV(pipeline, param_grid, scoring=get_scorer("kendall_tau"))
>>> search.fit(X, y).best_params_
{'kneighborslabelranker__n_neighbors': 1}

:ref:`sklearn_integration` describes how the rankers work with these and other tools
of scikit-learn, such as metadata routing and pandas data frames.

Next steps
----------

The :ref:`user guide <user_guide>` describes the representation of the rankings,
the estimators and the metrics, the :ref:`API reference <api_ref>` lists every
class and function, and the :ref:`examples <general_examples>` show them at work.
