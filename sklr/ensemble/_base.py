"""Base class for ensemble-based estimators."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC, abstractmethod
from copy import deepcopy
from numbers import Integral

# Third party
import numpy as np

# Local application
from ..base import BaseEstimator, MetaEstimatorMixin
from ..utils.validation import check_random_state


# =============================================================================
# Constants
# =============================================================================

# Maximum random number that
# can be used for seeding
MAX_RAND_SEED = np.iinfo(np.int32).max


# =============================================================================
# Methods
# =============================================================================

def _set_random_states(estimator, random_state=None):
    """Sets fixed random_state parameters for an estimator.

    Find all parameters ending ``random_state`` and sets them to integers
    derived from ``random_state``.

    Notes
    -----
    This does not necessarily set *all* ``random_state`` attributes that
    control an estimator's randomness, only those accessible through
    ``estimator.get_hyperparams()``.
    """
    # Obtain the random state
    random_state = check_random_state(random_state)

    # Initialize the dictionary where the
    # random states are going to be put on.
    # In case that the estimator does not
    # needed random state, this dictionary
    # will be empty and the process of
    # setting the seed skipped
    to_set = {}

    # Introduce the random state for
    # this estimator in the dictionary
    # (only if it is supported)
    for key in sorted(estimator.get_hyperparams(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(MAX_RAND_SEED)

    # Set the random state
    # in the estimator
    if to_set:
        estimator.set_hyperparams(**to_set)


def _indexes_to_mask(indexes, mask_length):
    """Convert list of indices to boolean mask."""
    # Initialize the mask of boolean values
    mask = np.zeros(mask_length, dtype=np.bool)

    # Set "True" in the given indexes
    mask[indexes] = True

    # Return the built mask
    return mask


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Base ensemble
# =============================================================================
class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, ABC):
    """Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Hyperparameters
    ---------------
    base_estimator : {None, object}, optional (default=None)
        The base estimator from which the ensemble is built.

    n_estimators : int
        The number of estimators in the ensemble.

    estimator_hyperparams : list of str
        The list of attributes to use as hyperparameters when instantiating a
        new base estimator.
        If none are given, default hyperparameters are used.

    Attributes
    ----------
    base_estimator_ : BaseEstimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of BaseEstimator
        The collection of fitted base estimators.
    """

    # Overwrite _required_parameters from
    # MetaEstimatorMixin to an empty list,
    # since the "estimator" object is not required
    _required_parameters = []

    @abstractmethod
    def __init__(self, base_estimator, n_estimators=10,
                 estimator_hyperparams=tuple()):
        """Constructor."""
        # Initialize the hyperparameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_hyperparams = estimator_hyperparams

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator hyperparameter,
        set the `base_estimator_` attribute."""
        # Check that the number of estimators
        # is an integer type greater than zero
        if not isinstance(self.n_estimators, (Integral, np.integer)):
            raise TypeError("n_estimators must be an integer. "
                            "Got {}."
                            .format(type(self.n_estimators)))
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero. "
                             "Got {}."
                             .format(self.n_estimators))

        # Check the base estimator, initializing to
        # the default one when it is not provided
        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        # Copy the base estimator to avoid
        # that the original one is modified
        estimator = deepcopy(self.base_estimator_)

        # Set the required hyperparameters
        # in the built estimator
        estimator.set_hyperparams(**{
                p: getattr(self, p) for p in self.estimator_hyperparams
            })

        # Set the random state in the estimator.
        # In fact, if the seed is not provided,
        # then, setting the random state is not necessary
        if random_state is not None:
            _set_random_states(estimator, random_state)

        # Include the estimator in the
        # list of estimators (if specified)
        if append:
            self.estimators_.append(estimator)

        # Return the made estimator
        return estimator
