"""Base classes for all estimators."""


# =============================================================================
# Imports
# =============================================================================

# Local application
from .consensus import RankAggregationAlgorithm
from .metrics import tau_score, tau_x_score


# =============================================================================
# Classes
# =============================================================================

class LabelRankerMixin:
    """Mixin class for all :term:`label rankers` in scikit-lr."""

    _estimator_type = "label_ranker"
    _rank_algorithm = RankAggregationAlgorithm.get_algorithm("borda_count")

    def score(self, X, Y, sample_weight=None):
        """Return the mean Kendall rank correlation coefficient
        :math:`\\tau` on the given test data and rankings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test samples.

        Y : array-like of shape (n_samples, n_classes)
            The true rankings for ``X``.

        sample_weight : array-like of shape (n_samples,), default=None
            The sample weights. If ``None``, then samples are equally
            weighted.

        Returns
        -------
        score : float
            The mean Kendall rank correlation coefficient :math:`\\tau`
            of ``self.predict(X)`` with respect to ``Y``.
        """
        return tau_score(Y, self.predict(X), sample_weight)


class PartialLabelRankerMixin:
    """Mixin class for all :term:`partial label rankers` in scikit-lr."""

    _estimator_type = "partial_label_ranker"
    _rank_algorithm = RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2")

    def score(self, X, Y, sample_weight=None):
        """Return the mean Kendall rank correlation coefficient
        :math:`\\tau_X` on the given test data and rankings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test samples.

        Y : array-like of shape (n_samples, n_classes)
            The true rankings for ``X``.

        sample_weight : array-like of shape (n_samples,), default=None
            The sample weights. If ``None``, then samples are equally
            weighted.

        Returns
        -------
        score : float
            The mean Kendall rank correlation coefficient :math:`\\tau_X`
            of ``self.predict(X)`` with respect to ``Y``.
        """
        return tau_x_score(Y, self.predict(X), sample_weight)


class TransformerMixin:
    """Mixin class for all transformers in scikit-lr."""

    def fit_transform(self, Y, **fit_params):
        """Fit to data, then transform it.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_classes)
            The target rankings.

        **fit_params : dict
            The additional fit parameters.

        Returns
        -------
        Yt : array-like of shape (n_samples, n_classes)
            The transformed array.
        """
        return self.fit(Y, **fit_params).transform(Y)


# =============================================================================
# Functions
# =============================================================================

def is_label_ranker(estimator):
    """Return ``True`` if the given estimator is a :term:`label ranker`.

    Parameters
    ----------
    estimator : object
        The estimator object to test.

    Returns
    -------
    out : bool
        ``True`` if ``estimator`` is a label ranker and ``False`` otherwise.
    """
    return estimator._estimator_type == "label_ranker"


def is_partial_label_ranker(estimator):
    """Return ``True`` if the given estimator is a :term:`partial label
    ranker`.

    Parameters
    ----------
    estimator : object
        The estimator object to test.

    Returns
    -------
    out : bool
        ``True`` if ``estimator`` is a partial label ranker and ``False``
        otherwise.
    """
    return estimator._estimator_type == "partial_label_ranker"
