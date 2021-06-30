"""Forest of tree-based ensemble methods."""


# =============================================================================
# Imports
# =============================================================================

# Standard
import numpy as np
from abc import ABCMeta

# Third party
from sklearn.ensemble._forest import BaseForest

# Local application
from ..base import LabelRankerMixin, PartialLabelRankerMixin
from ..tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker


# =============================================================================
# Classes
# =============================================================================

class ForestLabelRanker(LabelRankerMixin, BaseForest, metaclass=ABCMeta):

    def _set_oob_score(self, X, Y):
        raise NotImplementedError("")

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
                 class_weight=None,
                 max_samples=None):
        """Constructor."""
        super(ForestLabelRanker, self).__init__(base_estimator,
                                                n_estimators,
                                                estimator_params=estimator_params,
                                                bootstrap=bootstrap,
                                                oob_score=oob_score,
                                                n_jobs=n_jobs,
                                                random_state=random_state,
                                                verbose=verbose,
                                                warm_start=warm_start,
                                                class_weight=None,
                                                max_samples=max_samples)
        

    def predict(self, X):
        aggregate = self._rank_algorithm.aggregate
        n_samples = X.shape[0]
        Y = np.array([estimator.predict(X) for estimator in self.estimators_])
        #print(Y[:, 0])
        Y = [aggregate(Y[:, sample]) for sample in range(n_samples)]

        return np.array(Y)


class RandomForestLabelRanker(ForestLabelRanker):

    base_estimator = DecisionTreeLabelRanker(random_state=None)
    estimator_params = (
                "criterion",
                "max_depth",
                "min_samples_split",
                #"min_samples_leaf",
                #"min_weight_fraction_leaf",
                "max_features",
                #"max_leaf_nodes",
                #"min_impurity_decrease",
                "random_state",
                #"ccp_alpha",
            )

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
        super(RandomForestLabelRanker, self).__init__(self.base_estimator,
                                                      n_estimators,
                                                      estimator_params=self.estimator_params,
                                                      bootstrap=bootstrap,
                                                      oob_score=oob_score,
                                                      n_jobs=n_jobs,
                                                      random_state=random_state,
                                                      verbose=verbose,
                                                      warm_start=warm_start,
                                                      max_samples=max_samples)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # self.min_samples_leaf = min_samples_leaf
        # self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        # self.max_leaf_nodes = max_leaf_nodes
        # self.min_impurity_decrease = min_impurity_decrease
        # self.ccp_alpha = ccp_alpha
