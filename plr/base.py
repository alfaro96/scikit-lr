"""Base classes for all estimators."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import abstractmethod
from collections import defaultdict
from inspect import signature

# Third party
import numpy as np

# Local application
from .consensus import RankAggregationAlgorithm
from .metrics.label_ranking import tau_score
from .metrics.partial_label_ranking import tau_x_score
from .utils.ranking import (
    check_label_ranking_targets, check_partial_label_ranking_targets,
    type_of_targets)
from .utils.validation import (
    check_array, check_consistent_length, check_is_fitted,
    check_X_Y, _check_sample_weight,
    has_fit_parameter)


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Base Estimator
# =============================================================================
class BaseEstimator:
    """Base class for all estimators in plr.

    Notes
    -----
    All estimators should specify all the hyperparameters that
    can be set at the class level in their ``__init__`` as
    explicit keyword arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_hyperparam_names(cls):
        """Get the hyperparameters names for the estimator."""
        # Introspect the constructor arguments to
        # find the hyperparameters of the model
        init_signature = signature(
            getattr(cls.__init__, "deprecated_original", cls.__init__))

        # Consider the constructor argurments excluding "self"
        hyperparams = [
            hyperparam for hyperparam in init_signature.parameters.values()
            if (hyperparam.name != "self" and
                hyperparam.kind != hyperparam.VAR_KEYWORD)
        ]

        # Check that the hyperparameters are
        # given as explicit keyword arguments
        for hyperparam in hyperparams:
            if hyperparam.kind == hyperparam.VAR_POSITIONAL:
                raise RuntimeError("plr estimators should always "
                                   "specify their parameters in the signature "
                                   "of their __init__ (no varargs). "
                                   "{} with constructor {} does not "
                                   "follow this convention."
                                   .format(cls, init_signature))

        # Extract and sort argument names excluding "self"
        return sorted([hyperparam.name for hyperparam in hyperparams])

    def get_hyperparams(self, deep=True):
        """Get the hyperparameters for this estimator.

        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the hyperparameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        hyperparams: dict of (string, any)
            Hyperparameter names mapped to their values.
        """
        # Initialize the dictionary with the hyperparameters
        hyperparams = dict()

        # Get the hyperparameter names mapped to their values
        for key in self._get_hyperparam_names():
            # Obtain the value of the hyperparameter
            value = getattr(self, key)
            # Get the nested hyperparameters for the contained sub-estimators
            if deep and hasattr(value, "get_hyperparams"):
                # Update the dictionary
                hyperparams.update(
                    (key + "__" + deep_key, deep_value)
                    for (deep_key, deep_value)
                    in value.get_hyperparams().items())
            # Set the value into the dictionary
            hyperparams[key] = value

        # Return the hyperparameters mapped to their values
        return hyperparams

    def set_hyperparams(self, **hyperparams):
        """Set the hyperparameters for this estimator.

        This method works on simple estimators as well as on nested
        objects (such as ensembles). The latter have parameters of the form
        ``<component>__<parameter>`` so that it is possible to update each
        component of a nested object.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If one of the specified hyperparameters is invalid.
        """
        # Simple optimization to gain speed (inspect is slow)
        if not hyperparams:
            return self

        # Obtain the valid hyperparameters for the estimator
        # (and the contained sub-estimators)
        valid_hyperparams = self.get_hyperparams(deep=True)

        # Initialize a dictionary for the hyperparameters
        # of the nested estimators to set all them at once
        nested_hyperparams = defaultdict(dict)

        # Set the values of the hyperparameters
        for (key, value) in hyperparams.items():
            # Obtain the keys for the estimators an
            # the sub-keys for the nested estimators
            (key, delim, sub_key) = key.partition("__")
            # Raise an error when the key is invalid
            if key not in valid_hyperparams:
                raise ValueError("Invalid hyperparameter {} for estimator {}. "
                                 "Check the list of available hyperparameters "
                                 "with `estimator.estimator.get_hyperparams()."
                                 "keys()`."
                                 .format(key, self))
            # Set the hyperparameters for the nested estimators
            if delim:
                nested_hyperparams[key][sub_key] = value
            # Set the hyperparameters for the estimator
            else:
                setattr(self, key, value)
                valid_hyperparams[key] = value

        # Recursively set the values of the
        # hypeparameters for the nested estimators
        for (key, sub_hyperparams) in nested_hyperparams.items():
            valid_hyperparams[key].set_hyperparams(**sub_hyperparams)

        # Return the estimator with the new hyperparameters
        return self


# =============================================================================
# Label Ranker Mixin
# =============================================================================
class LabelRankerMixin:
    """Mixin class for all Label Rankers in plr."""

    # Estimator type
    _estimator_type = "label_ranker"

    # Rank Aggregation algorithm
    _rank_algorithm = RankAggregationAlgorithm.get_algorithm("borda_count")

    def _validate_training_data(self, X, Y, sample_weight=None):
        """Validate the training samples and rankings for the Label Ranker."""
        # Check the training samples and rankings
        (X, Y) = check_X_Y(X, Y)

        # Check that the training rankings are for Label Ranking
        check_label_ranking_targets(Y)

        # Check whether the Label Ranker supports sample weighting
        support_sample_weighting = has_fit_parameter(self, "sample_weight")

        # Check the sample weights if the
        # Label Ranker supports sample weighting
        if support_sample_weighting:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Determine the output settings
        (self.n_samples_, self.n_features_) = X.shape
        self.n_classes_ = Y.shape[1]

        # Return the validated training samples and rankings and, also, the
        # sample weights if the Label Ranker supports sample weighting
        if support_sample_weighting:
            return (X, Y, sample_weight)
        else:
            return (X, Y)

    def _validate_test_data(self, X):
        """Validate the test samples for the Label Ranker."""
        # Check the test samples
        X = check_array(X, dtype=np.float64)

        # Obtain the number of features
        n_features = X.shape[1]

        # Check that the number of features of the
        # training and test samples are the same
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {} and "
                             "input n_features is {}."
                             .format(self.n_features_, n_features))

        # Return the validated test samples
        return X

    def score(self, X, Y, sample_weight=None):
        """Returns the mean Tau score on the given test data and rankings.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.

        Y : np.ndarray of shape (n_samples, n_classes)
            True rankings for X.

        sample_weight: {None, np.ndarray} of shape (n_samples,),
                optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Mean Tau score of self.predict(X) w.r.t. Y.
        """
        return tau_score(Y, self.predict(X), sample_weight)


# =============================================================================
# Partial Label Ranker Mixin
# =============================================================================
class PartialLabelRankerMixin:
    """Mixin class for all Partial Label Rankers in plr."""

    # Estimator type
    _estimator_type = "partial_label_ranker"

    # Rank Aggregation algorithm
    _rank_algorithm = RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2")

    def _validate_training_data(self, X, Y, sample_weight=None):
        """Validate the training samples and
        rankings for the Partial Label Ranker."""
        # Check the training samples and rankings
        (X, Y) = check_X_Y(X, Y)

        # Check that the training rankings are for Partial Label Ranking
        check_partial_label_ranking_targets(Y)

        # Check whether the Partial Label Ranker supports sample weighting
        support_sample_weighting = has_fit_parameter(self, "sample_weight")

        # Check the sample weights if the Partial
        # Label Ranker supports sample weighting
        if support_sample_weighting:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Determine the output settings
        (self.n_samples_, self.n_features_) = X.shape
        self.n_classes_ = Y.shape[1]

        # Return the validated training samples and rankings and, also, the
        # sample weights if the Partial Label Ranker supports sample weighting
        if support_sample_weighting:
            return (X, Y, sample_weight)
        else:
            return (X, Y)

    def _validate_test_data(self, X):
        """Validate the test samples for the Partial Label Ranker."""
        # Check the test samples
        X = check_array(X, dtype=np.float64)

        # Obtain the number of features
        n_features = X.shape[1]

        # Check that the number of features of the
        # training and test samples are the same
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {} and "
                             "input n_features is {}."
                             .format(self.n_features_, n_features))

        # Return the validated test samples
        return X

    def score(self, X, Y, sample_weight=None):
        """Returns the mean Tau-x score on the given test data and rankings.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.

        Y : np.ndarray of shape (n_samples, n_classes)
            True rankings for X.

        sample_weight : {None, np.ndarray} of shape (n_samples,),
                optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Mean Tau-x score of self.predict(X) w.r.t. Y.
        """
        return tau_x_score(Y, self.predict(X), sample_weight)


# =============================================================================
# Transformer Mixin
# =============================================================================
class TransformerMixin:
    """Mixin class for all transformers in plr."""

    def _validate_training_data(self, Y):
        """Validate the training rankings for the transformer."""
        # Obtain the target types
        self.target_types_ = type_of_targets(Y)

        # Determine the output settings
        (self.n_samples_, self.n_classes_) = Y.shape

        # Return the validated training rankings
        return Y

    def _validate_test_data(self, Y):
        """Validate the test rankings for the transformer."""
        # Obtain the target types
        target_types = type_of_targets(Y)

        # Check the target types
        if not target_types.intersection(self.target_types_):
            raise ValueError("The type of targets of the input rankings "
                             "are {} while the allowed type of targets are {}."
                             .format(
                                 sorted(list(target_types)),
                                 sorted(list(self.target_types_))))

        # Check the number of classes
        if self.n_classes_ != Y.shape[1]:
            raise ValueError("Y has {} classes per sample, expected {}."
                             .format(Y.shape[1], self.n_classes_))

        # Return the validated test rankings
        return Y

    def fit_transform(self, Y, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to Y with optional parameters
        fit_params and returns a transformed version of Y.

        Parameters
        ----------
        Y : np.ndarray of shape (n_samples, n_classes)
            Training rankings.

        Returns
        -------
        Yt : np.ndarray of shape (n_samples, n_classes)
            Transformed array.
        """
        return self.fit(Y, **fit_params).transform(Y)


# =============================================================================
# Meta Estimator Mixin
# =============================================================================
class MetaEstimatorMixin:
    """Mixin class for all meta estimators in plr."""

    # Required parameters for the estimators
    _required_parameters = ["estimator"]


# =============================================================================
# Methods
# =============================================================================

def is_label_ranker(estimator):
    """Returns True if the given estimator is (probably) a Label Ranker.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a Label Ranker and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "label_ranker"


def is_partial_label_ranker(estimator):
    """Returns True if the given estimator
    is (probably) a Partial Label Ranker.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a Partial Label Ranker and False otherwise.
    """
    return (
        getattr(estimator, "_estimator_type", None) == "partial_label_ranker")
