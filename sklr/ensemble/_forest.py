"""This module includes forest of tree-based ensemble methods."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABCMeta

# Third party
from sklearn.ensemble._forest import BaseForest

# Local application
from ._base import _predict_ensemble
from ..base import LabelRankerMixin, PartialLabelRankerMixin
from ..tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker


# =============================================================================
# Classes
# =============================================================================

class ForestLabelRanker(LabelRankerMixin, BaseForest, metaclass=ABCMeta):
    """Base class for forest of trees-based label rankers."""

    def __init__(self,
                 base_estimator,
                 n_estimators=100,
                 *,
                 estimator_params=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_samples=None):
        """Constructor."""
        super(ForestLabelRanker, self).__init__(base_estimator,
                                                n_estimators,
                                                estimator_params=estimator_params,  # noqa
                                                bootstrap=bootstrap,
                                                oob_score=oob_score,
                                                n_jobs=n_jobs,
                                                random_state=random_state,
                                                verbose=verbose,
                                                warm_start=warm_start,
                                                class_weight=None,
                                                max_samples=max_samples)

    def predict(self, X):
        """"""
        return _predict_ensemble(self, X, None)


class RandomForestLabelRanker(ForestLabelRanker):
    """A Random Forest :term:`label ranker`.

    A Random Forest is a meta estimator that fits a number of decision tree
    label rankers on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.

    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to build
    each tree.

    Read more in the :ref:`User Guide <forest>`.


    """
    
    

    def __init__(self,
                 n_estimators=100,
                 *,
                 criterion="mallows",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None):
        """Constructor."""
        #
        base_estimator = DecisionTreeLabelRanker(max_depth=None)

        estimator_params = ("criterion",
                            "max_depth",
                            "min_samples_split",
                            "min_samples_leaf",
                            "min_weight_fraction_leaf",
                            "max_features",
                            "max_leaf_nodes",
                            "min_impurity_decrease",
                            "ccp_alpha",
                            "random_state")

        super(RandomForestLabelRanker, self).__init__(base_estimator,
                                                      n_estimators,
                                                      estimator_params=estimator_params,  # noqa
                                                      bootstrap=bootstrap,
                                                      oob_score=oob_score,
                                                      n_jobs=n_jobs,
                                                      random_state=random_state,  # noqa
                                                      verbose=verbose,
                                                      warm_start=warm_start,
                                                      max_samples=max_samples)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
