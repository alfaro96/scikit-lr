"""Base classes for all estimators."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from collections import defaultdict
from inspect import signature

# Third party
import numpy as np

# Local application
from .consensus import RankAggregationAlgorithm
from .metrics import tau_score, tau_x_score
from .utils import (check_array, check_is_fitted, check_sample_weight,
                    check_X_Y, has_fit_parameter)


# =============================================================================
# Classes
# =============================================================================

class BaseEstimator:
    """Base class for all estimators in scikit-lr.

    Notes
    -----
    All estimators should specify all the hyperparameters that can be set at
    the class level in their ``__init__`` as explicit keyword arguments (no
    ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_hyperparam_names(cls, raise_varargs):
        """Get the hyperparameter names for this estimator."""
        # Fetch the constructor and introspect the arguments
        # to find the model hyperparameters to represent
        init = signature(getattr(cls, "__init__"))
        hyperparams = init.parameters

        for key in hyperparams:
            if ((hyperparams[key].kind == hyperparams[key].VAR_POSITIONAL or
                    hyperparams[key].kind == hyperparams[key].VAR_KEYWORD) and
                    raise_varargs):
                raise RuntimeError("Scikit-lr estimators should always "
                                   "specify their hyperparameters in the "
                                   "signature of their __init__ (no varargs). "
                                   "{0} with constructor {1} does not follow "
                                   "this convention.".format(cls, init))

        # Extract and sort the hyperparameters (excluding "self")
        return sorted([key for key in hyperparams if key != "self"])

    def get_hyperparams(self, deep=True):
        """Get the hyperparameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If ``True``, will return the hyperparameters for this
            estimator and contained sub-objects that are estimators.

        Returns
        -------
        hyperparams : dict
            The hyperparameter names mapped to their values.
        """
        hyperparams = defaultdict(dict)

        for key in self._get_hyperparam_names(raise_varargs=True):
            # "None" will be returned for the
            # hyperparameters that cannot be
            # retrieved as instance attribute
            value = getattr(self, key, None)
            hyperparams[key] = value
            # Get the hyperparameters for the sub-object
            if deep and hasattr(value, "get_hyperparams"):
                deep_hyperparams = value.get_hyperparams(deep=True)
                # Use a generator expression to set all the
                # hyperparameters of the sub-object at once
                hyperparams.update((key + "__" + deep_key,
                                    deep_hyperparams[deep_key])
                                   for deep_key in deep_hyperparams)

        return dict(hyperparams)

    def set_hyperparams(self, **hyperparams):
        """Set the hyperparameters for this estimator.

        The method works on simple estimators as well as on nested objects.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it is possible to update each component of a nested object.

        Parameters
        ----------
        **hyperparams : dict
            The estimator hyperparameters.

        Returns
        -------
        self : object
            The estimator instance.
        """
        # Use a default dictionary for the nested hyperparameters
        # to automatically create the keys when they are accessed
        valid_hyperparams = self.get_hyperparams(deep=True)
        nested_hyperparams = defaultdict(dict)

        for key in hyperparams:
            # Obtain the keys for the estimators and
            # the sub-keys for the nested estimators
            value = hyperparams[key]
            (key, delim, sub_key) = key.partition("__")
            # Raise an informative message when the hyperparameter is invalid
            if key not in valid_hyperparams:
                raise ValueError("Invalid hyperparameter {0} for estimator "
                                 "{1}. Check the available hyperparameters "
                                 "with `estimator.get_hyperparams().keys()`."
                                 .format(key, self.__class__.__name__))
            # Set the estimator hyperparameter and put
            # the value in the (nested) dictionary
            if delim:
                nested_hyperparams[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_hyperparams[key] = value

        for key in nested_hyperparams:
            # Transform the nested hyperparameters of the
            # estimator to hyperparameters of the sub-estimator
            sub_hyperparams = nested_hyperparams[key]
            valid_hyperparams[key].set_hyperparams(**sub_hyperparams)

        return self

    def _validate_train_data(self, X, Y, sample_weight=None):
        """Validate the training samples and rankings for this
        estimator and, if possible, validate the sample weights."""
        (X, Y) = check_X_Y(X, Y)

        # Check that the training rankings can be managed by the Rank
        # Aggregation algorithm of this estimator, since they are not
        # tested if the underlying fast methods are employed directly
        self._rank_algorithm.check_targets(Y)

        support_sample_weighting = has_fit_parameter(self, "sample_weight")

        if support_sample_weighting:
            # Does not matter whether the samples weights are
            # checked with respect to the training samples or
            # the training rankings, since both has been checked
            sample_weight = check_sample_weight(sample_weight, X)

        # Determine the output shape to validate the
        # test data when predicting with the estimator
        (self.n_samples_in_, self.n_features_in_) = X.shape
        (_, self.n_classes_in_) = Y.shape

        return (X, Y, sample_weight) if support_sample_weighting else (X, Y)

    def _validate_train_rankings(self, Y):
        """Validate the training rankings for this estimator."""
        Y = check_array(Y, dtype=np.int64)

        # Determine the output shape to validate the test
        # rankings when transforming with the estimator
        (self.n_samples_in_, self.n_classes_in_) = Y.shape

        return Y

    def _validate_test_data(self, X):
        """Validate the test samples for this estimator."""
        check_is_fitted(self)

        X = check_array(X, dtype=np.float64)

        if self.n_features_in_ != X.shape[1]:
            raise ValueError("Number of features of the model must match "
                             "the input. Model number of features is {0} "
                             "and input number of features is {1}."
                             .format(self.n_features_in_, X.shape[1]))

        return X

    def _validate_test_rankings(self, Y):
        """Validate the test rankings for this estimator."""
        check_is_fitted(self)

        Y = check_array(Y, dtype=np.int64)

        if self.n_classes_in_ != Y.shape[1]:
            raise ValueError("Number of classes of the model must match "
                             "the input. Model number of classes is {0} "
                             "and input number of classes is {1}."
                             .format(self.n_classes_in_, Y.shape[1]))

        return Y


class LabelRankerMixin:
    """Mixin class for all Label Rankers in scikit-lr."""

    _estimator_type = "label_ranker"
    _rank_algorithm = RankAggregationAlgorithm.get_algorithm("borda_count")

    def score(self, X, Y, sample_weight=None):
        """Return the mean tau score on the given test data and rankings.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), dtype=np.float64
            The test samples.

        Y : ndarray of shape (n_samples, n_classes), dtype=np.int64
            The true rankings for ``X``.

        sample_weight : ndarray of shape (n_samples,), dtype=np.float64, \
                default=None
            The sample weights. If ``None``, then samples are equally weighted.

        Returns
        -------
        score : float
            The mean tau score of ``self.predict(X)`` with respect to ``Y``.
        """
        return tau_score(Y, self.predict(X), sample_weight)


class PartialLabelRankerMixin:
    """Mixin class for all Partial Label Rankers in scikit-lr."""

    _estimator_type = "partial_label_ranker"
    _rank_algorithm = RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2")

    def score(self, X, Y, sample_weight=None):
        """Return the mean tau-x score on the given test data and rankings.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), dtype=np.float64
            The test samples.

        Y : ndarray of shape (n_samples, n_classes), dtype=np.int64
            The true rankings for ``X``.

        sample_weight : ndarray of shape (n_samples,), dtype=np.float64, \
                default=None
            The sample weights. If ``None``, then samples are equally weighted.

        Returns
        -------
        score : float
            The mean tau-x score of ``self.predict(X)`` with respect to ``Y``.
        """
        return tau_x_score(Y, self.predict(X), sample_weight)


class TransformerMixin:
    """Mixin class for all transformers in scikit-lr."""

    def fit_transform(self, Y, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to ``Y`` with optional parameters
        ``fit_params`` and returns a transformed version of ``Y``.

        Parameters
        ----------
        Y : ndarray of shape (n_samples, n_classes), dtype=np.int64
            The training rankings.

        **fit_params : dict
            The additional fit parameters.

        Returns
        -------
        Yt : ndarray of shape (n_samples, n_classes), dtype=np.float64
            The transformed array.
        """
        return self.fit(Y, **fit_params).transform(Y)


class MetaEstimatorMixin:
    """Mixin class for all meta-estimators in scikit-lr."""

    _required_parameters = ["estimator"]


# =============================================================================
# Methods
# =============================================================================

def is_label_ranker(estimator):
    """Return ``True`` if the given estimator is a Label Ranker.

    Parameters
    ----------
    estimator : object
        The estimator object to test.

    Returns
    -------
    out : bool
        ``True`` if ``estimator`` is a Label Ranker and ``False`` otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "label_ranker"


def is_partial_label_ranker(estimator):
    """Return ``True`` if the given estimator is a Partial Label Ranker.

    Parameters
    ----------
    estimator : object
        The estimator object to test.

    Returns
    -------
    out : bool
        ``True`` if ``estimator`` is a Partial Label Ranker and ``False``
        otherwise.
    """
    return (
        getattr(estimator, "_estimator_type", None) == "partial_label_ranker")
