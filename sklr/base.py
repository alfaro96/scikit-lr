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
    """Base class for all estimators in scikit-lr.

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

        # Consider the constructor arguments (excluding "self")
        hyperparams = [
            hyperparam for hyperparam in init_signature.parameters.values()
            if (hyperparam.name != "self" and
                hyperparam.kind != hyperparam.VAR_KEYWORD)
        ]

        # Check that the hyperparameters are given as
        # explicit keyword arguments, raising the proper
        # error when they are given as "*args" or "**kwargs"
        for hyperparam in hyperparams:
            if hyperparam.kind == hyperparam.VAR_POSITIONAL:
                raise RuntimeError("scikit-lr estimators should always "
                                   "specify their parameters in the signature "
                                   "of their __init__ (no varargs). "
                                   "{} with constructor {} does not "
                                   "follow this convention."
                                   .format(cls, init_signature))

        # Extract and sort the hyperparameters names
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
        hyperparams : dict
            Hyperparameter names mapped to their values.
        """
        # Initialize an empty dictionary that will hold
        # the hyperparameters names mapped to their values
        hyperparams = dict()

        # Fill the previous dictionary
        for key in self._get_hyperparam_names():
            # Obtain the value of the hyperparameter
            # from its name, that has been given as key
            value = getattr(self, key)
            # Get the nested hyperparameters
            # for the contained sub-estimators
            if deep and hasattr(value, "get_hyperparams"):
                # Update the dictionary
                hyperparams.update(
                    (key + "__" + deep_key, deep_value)
                    for (deep_key, deep_value)
                    in value.get_hyperparams().items())
            # Set the value into the dictionary
            hyperparams[key] = value

        # Return the hyperparameters names mapped to their values
        return hyperparams

    def set_hyperparams(self, **hyperparams):
        """Set the hyperparameters for this estimator.

        This method works on simple estimators as well as on nested
        objects (such as ensembles). The latter have parameters of the form
        ``<component>__<parameter>`` so that it is possible to update each
        component of a nested object.

        Parameters
        ----------
        **hyperparams : dict
            Estimator hyperparameters.

        Returns
        -------
        self : object
            Estimator instance.

        Raises
        ------
        ValueError
            If one of the specified hyperparameters is invalid.
        """
        # Simple optimization to gain speed (inspect is slow) by directly
        # returning this object when no hyperparameters are given
        if not hyperparams:
            return self

        # Obtain the valid hyperparameters for the estimator
        # (and the contained sub-objects that are estimators)
        valid_hyperparams = self.get_hyperparams(deep=True)

        # Initialize a dictionary for the hyperparameters
        # of the nested estimators to set all them at once
        nested_hyperparams = defaultdict(dict)

        # Set the hyperparameters on this estimator,
        # raising the proper error when a given
        # hyperparameter is not of this estimator
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

        # Return this estimator instances with
        # the new hyperparameters properly set
        return self


# =============================================================================
# Label Ranker Mixin
# =============================================================================
class LabelRankerMixin:
    """Mixin class for all Label Rankers in scikit-lr."""

    # Estimator type
    _estimator_type = "label_ranker"

    # Rank Aggregation algorithm
    _rank_algorithm = RankAggregationAlgorithm.get_algorithm("borda_count")

    def _validate_training_data(self, X, Y, sample_weight=None):
        """Validate the training samples and rankings for this Label
        Ranker and, if possible, also validate the sample weights."""
        # Check the training samples and rankings
        (X, Y) = check_X_Y(X, Y)

        # Check that the training rankings
        # are for the Label Ranking problem
        check_label_ranking_targets(Y)

        # Check whether this Label Ranker supports
        # sample weighting (to also validate them)
        support_sample_weighting = has_fit_parameter(self, "sample_weight")

        # Check the sample weights if the
        # Label Ranker supports sample weighting
        if support_sample_weighting:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Determine the output settings
        (self.n_samples_, self.n_features_) = X.shape
        self.n_classes_ = Y.shape[1]

        # Return the validated training samples and rankings and,
        # also, the sample weights if the Label Ranker supports them
        if support_sample_weighting:
            return (X, Y, sample_weight)
        else:
            return (X, Y)

    def _validate_test_data(self, X):
        """Validate the test samples for this Label Ranker."""
        # Check the format of the test samples to ensure that
        # they are properly formatted for downstream estimators
        X = check_array(X, dtype=np.float64)

        # Determine the output setting of the test data
        n_features = X.shape[1]

        # Ensure that the the number of features of the
        # training and test samples are the same, raising
        # the proper error when this condition does not hold
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {} and "
                             "input n_features is {}."
                             .format(self.n_features_, n_features))

        # Return the validated test samples
        return X

    def score(self, X, Y, sample_weight=None):
        """Return the mean Tau score on the given test data and rankings.

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
    """Mixin class for all Partial Label Rankers in scikit-lr."""

    # Estimator type
    _estimator_type = "partial_label_ranker"

    # Rank Aggregation algorithm
    _rank_algorithm = RankAggregationAlgorithm.get_algorithm("bpa_lia_mp2")

    def _validate_training_data(self, X, Y, sample_weight=None):
        """Validate the training samples and
        rankings for the Partial Label Ranker."""
        # Check the training samples and rankings
        (X, Y) = check_X_Y(X, Y)

        # Check that the training rankings
        # are for the Partial Label Ranking problem
        check_partial_label_ranking_targets(Y)

        # Check whether this Partial Label Ranker supports
        # sample weighting (to also validate them)
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
        # Check the format of the test samples to ensure that
        # they are properly formatted for downstream estimators
        X = check_array(X, dtype=np.float64)

        # Determine the output setting of the test data
        n_features = X.shape[1]

        # Ensure that the the number of features of the
        # training and test samples are the same, raising
        # the proper error when this condition does not hold
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {} and "
                             "input n_features is {}."
                             .format(self.n_features_, n_features))

        # Return the validated test samples
        return X

    def score(self, X, Y, sample_weight=None):
        """Return the mean Tau-x score on the given test data and rankings.

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
    """Mixin class for all transformers in scikit-lr."""

    def _validate_training_rankings(self, Y):
        """Validate the training rankings for the transformer."""
        # Check the format of the training rankings, allowing
        # infinite values since some classes may be missed
        Y = check_array(Y, force_all_finite=False)

        # Obtain the target types of the training rankings to ensure
        # that, at least, one of them is the same that the target types
        # of the test rankings (passed in the transform method)
        self.target_types_ = type_of_targets(Y)

        # Determine the output settings
        (self.n_samples_, self.n_classes_) = Y.shape

        # Return the validated training rankings
        return Y

    def _validate_test_rankings(self, Y):
        """Validate the test rankings for the transformer."""
        # Check the format of the training rankings, allowing
        # infinite values since some classes may be missed
        Y = check_array(Y, force_all_finite=False)

        # Obtain the target types of the test rankings to ensure that
        # one of them is in the target types of the training rankings
        target_types = type_of_targets(Y)

        # Check the target types
        if not target_types.intersection(self.target_types_):
            raise ValueError("The type of targets of the input rankings "
                             "are {} while the allowed type of targets are {}."
                             .format(
                                 sorted(list(target_types)),
                                 sorted(list(self.target_types_))))

        # Check that the number of classes of the test rankings is
        # the same that the number of classes of the training rankings
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

        **fit_params : dict
            Additional fit parameters.

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
    """Mixin class for all meta estimators in scikit-lr."""

    # Required parameters for the meta-estimators
    _required_parameters = ["estimator"]


# =============================================================================
# Methods
# =============================================================================

def is_label_ranker(estimator):
    """Returns True if the given estimator
    is (probably) a Label Ranker.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a Label Ranker and False otherwise.
    """
    return (
        getattr(estimator, "_estimator_type", None) == "label_ranker")


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
