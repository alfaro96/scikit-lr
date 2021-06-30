"""Bagging meta-estimator."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
from sklearn.ensemble._bagging import BaseBagging

# Local application
from ..base import LabelRankerMixin, PartialLabelRankerMixin
from ..tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker


# =============================================================================
# Classes
# =============================================================================

class BaggingLabelRanker(LabelRankerMixin, BaseBagging):

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 *,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        """Constructor."""
        super(BaggingLabelRanker, self).__init__(base_estimator,
                                                 n_estimators,
                                                 max_samples=max_samples,
                                                 max_features=max_features,
                                                 bootstrap=bootstrap,
                                                 bootstrap_features=bootstrap_features,  # noqa
                                                 oob_score=oob_score,
                                                 warm_start=warm_start,
                                                 n_jobs=n_jobs,
                                                 random_state=random_state,
                                                 verbose=verbose)

    def _validate_estimator(self):
        """Check the estimator and set the `base_estimator_` attribute."""
        # The random state will be set by the ensemble
        estimator = DecisionTreeLabelRanker(random_state=None)

        super(BaggingLabelRanker, self)._validate_estimator(estimator)

    def _set_oob_score(self, X, Y):
        raise NotImplementedError("")

    def _validate_y(self, Y):
        """Validate the target rankings."""
        return Y

    def predict(self, X):
        aggregate = self._rank_algorithm.aggregate
        n_samples = X.shape[0]
        Y = np.array([estimator.predict(X) for estimator in self.estimators_])
        #print(Y[:, 0])
        Y = [aggregate(Y[:, sample]) for sample in range(n_samples)]

        return np.array(Y)
