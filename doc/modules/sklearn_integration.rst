.. _sklearn_integration:

Using the rankers with scikit-learn
===================================

The rankers of scikit-lr are scikit-learn estimators, so they work with the tools of
scikit-learn that take an estimator, such as pipelines, the model selection tools
and :func:`~sklearn.base.clone`, and they can be pickled. This section describes
what is specific to the rankers. The rest is described in the scikit-learn user
guide, which these sections link to.

Pipelines
---------

A ranker can be the last step of a :class:`~sklearn.pipeline.Pipeline`, whose
previous steps transform the features, as described in
:ref:`combining estimators <sklearn:combining_estimators>`. The transformers only
see the features, while the rankings are passed to the ranker as they are. For
instance, a :class:`~sklearn.compose.ColumnTransformer` can encode the categorical
features before a ranker, which needs numerical features, as in
:ref:`sphx_glr_auto_examples_model_selection_plot_data_frames_sample_weight.py`.
Scaling matters for the nearest neighbors rankers, which find the neighbors with a
distance between the features, but not for the trees, whose splits only compare
each feature with a threshold.

Model selection
---------------

The rankers are evaluated with :ref:`cross-validation <sklearn:cross_validation>`
and their parameters are tuned with the :ref:`parameter search
<sklearn:grid_search>` tools of scikit-learn, such as
:func:`~sklearn.model_selection.cross_validate` and
:class:`~sklearn.model_selection.GridSearchCV`. By default, they use the ``score``
method of the ranker, and the other metrics of scikit-lr are passed with their
scorers, as described in :ref:`scoring`:

>>> import numpy as np
>>> from scipy.stats import rankdata
>>> from sklearn.model_selection import GridSearchCV
>>> from sklr.metrics import get_scorer
>>> from sklr.tree import DecisionTreeLabelRanker
>>> rng = np.random.RandomState(0)
>>> X = rng.normal(size=(100, 2))
>>> y = rankdata(-X @ rng.normal(size=(2, 4)), axis=1)  # rankings of 4 labels
>>> search = GridSearchCV(
...     DecisionTreeLabelRanker(random_state=0),
...     {"max_depth": [1, 2, 4]},
...     scoring=get_scorer("neg_kendall_distance"),
... )
>>> search.fit(X, y).best_params_
{'max_depth': 4}

The splitters of scikit-learn, such as :class:`~sklearn.model_selection.KFold`,
only look at the samples, so they split the rankings as any target. Those that
stratify the samples, such as :class:`~sklearn.model_selection.StratifiedKFold`,
need classes, so they do not work with rankings.

Parallelism and randomness
--------------------------

The ``n_jobs`` parameter of the model selection tools fits the rankers in parallel,
as described in :ref:`parallelism <sklearn:parallelism>`. The ``n_jobs`` parameter of
the nearest neighbors rankers is independent: it sets the threads that search the
neighbors and aggregate their rankings within a single ranker.

The trees choose the features of each split at random, as in scikit-learn, so
their ``random_state`` must be fixed to an integer for the results to be
reproducible, as described in :ref:`randomness <sklearn:randomness>`. Then the
model selection tools give the same results with any number of jobs, because each
ranker only depends on its own ``random_state``.

.. _sklearn_integration_sample_weights:

Sample weights and metadata routing
-----------------------------------

The trees and the dummy rankers accept ``sample_weight`` in their ``fit`` method,
which weighs the training rankings, and every ranker accepts it in its ``score``
method, which weighs the score of each sample. The nearest neighbors rankers do not
accept it in ``fit``, as those of scikit-learn, since they weigh the neighbors with
their ``weights`` parameter.

When a ranker is inside a pipeline or a model selection tool, the weights are passed
to it with :ref:`metadata routing <sklearn:metadata_routing>`, which must be
enabled. The ranker then requests them with its ``set_fit_request`` and
``set_score_request`` methods, and each step of a pipeline that accepts them
states whether it wants them too:

>>> import sklearn
>>> from sklearn.model_selection import cross_validate
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> sample_weight = rng.uniform(size=100)
>>> with sklearn.config_context(enable_metadata_routing=True):
...     ranker = (
...         DecisionTreeLabelRanker(max_depth=4, random_state=0)
...         .set_fit_request(sample_weight=True)
...         .set_score_request(sample_weight=True)
...     )
...     scaler = StandardScaler().set_fit_request(sample_weight=False)
...     pipeline = make_pipeline(scaler, ranker)
...     results = cross_validate(
...         pipeline, X, y, params={"sample_weight": sample_weight}
...     )
>>> results["test_score"].round(2)
array([0.83, 0.57, 0.86, 0.86, 0.8 ])

The scorers of :func:`~sklr.metrics.get_scorer` request the weights with their own
``set_score_request`` method.

pandas data frames
------------------

The features may be a pandas ``DataFrame``, whose column names the rankers store in
their ``feature_names_in_`` attribute and check when they predict, as the estimators
of scikit-learn do. The rankings may also be a data frame with a column per label,
in which case the columns of the predicted rankings follow the order of its
columns, and the sample weights may be a ``Series``.

Ensembles
---------

The ensembles of scikit-learn combine the predictions of their estimators as
classes or real values, which is not how rankings are aggregated. Most of them
reject targets with several columns, such as
:class:`~sklearn.ensemble.VotingRegressor` and
:class:`~sklearn.ensemble.StackingRegressor`, but
:class:`~sklearn.ensemble.BaggingRegressor` averages the positions of the rankings
that its rankers predict, so its predictions are not rankings. None of them should
be used with the rankers.
