"""Weight boosting."""

import numpy as np
from sklearn.base import MultiOutputMixin
from sklearn.ensemble._weight_boosting import BaseWeightBoosting
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples
from ..base import LabelRankerMixin
from ..tree import DecisionTreeLabelRanker
from sklr.metrics import kendall_distance




# =============================================================================
# Classes
# =============================================================================

class AdaBoostLabelRanker(LabelRankerMixin, BaseWeightBoosting):

    def __init__(self,
                 base_estimator=None,
                 *,
                 n_estimators=50,
                 learning_rate=1.0,
                 random_state=None):

        super(AdaBoostLabelRanker, self).__init__(base_estimator,
                                                  n_estimators=n_estimators,
                                                  learning_rate=learning_rate,
                                                  random_state=random_state)

    def fit(self, X, Y, sample_weight=None):
        X, self._Y = self._validate_data(X, Y, multi_output=True)
        y = Y[:, 0]
        return super(AdaBoostLabelRanker, self).fit(X, y, sample_weight)

    def _validate_estimator(self):
        default = DecisionTreeLabelRanker(max_depth=3)
        return super(AdaBoostLabelRanker, self)._validate_estimator(default)

    def _boost(self, iboost, X, Y, sample_weight, random_state):
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        bootstrap_idx = random_state.choice(
            np.arange(_num_samples(X)),
            size=_num_samples(X),
            replace=True,
            p=sample_weight,
        )

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        _X = _safe_indexing(X, bootstrap_idx)
        _Y = _safe_indexing(self._Y, bootstrap_idx)
        estimator.fit(_X, _Y)
        Y_predict = estimator.predict(X)

        error_vect = np.array([kendall_distance(self._Y[sample, None], Y_predict[sample, None], normalize=True) for sample in range(X.shape[0])])
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]

        error_max = np.max(masked_error_vector)
        if error_max != 0:
            masked_error_vector /= error_max

        # Calculate the average loss
        estimator_error = np.sum(masked_sample_weight * masked_error_vector)

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1.0, 0.0

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1.0 - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1.0 / beta)

        if not iboost == self.n_estimators - 1:
            sample_weight[sample_mask] *= np.power(
                beta, (1.0 - masked_error_vector) * self.learning_rate
            )

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        limit = len(self.estimators_)
        aggregate = self._rank_algorithm.aggregate
        n_samples = X.shape[0]
        Y = np.array([estimator.predict(X) for estimator in self.estimators_])
        print(Y)
        print(self.estimator_weights_)
        Y = [aggregate(Y[:, sample], self.estimator_weights_[:limit]) for sample in range(n_samples)]

        return np.array(Y)
