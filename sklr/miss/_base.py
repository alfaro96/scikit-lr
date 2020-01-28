"""This module gathers base transformers to miss classes from rankings."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from numbers import Integral, Real

# Third party
import numpy as np

# Local application
from ._base_fast import miss_classes
from ._base_fast import STRATEGIES
from ..base import BaseEstimator, TransformerMixin
from ..utils.validation import check_is_fitted, check_random_state


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Simple misser
# =============================================================================
class SimpleMisser(BaseEstimator, TransformerMixin):
    """Transformer to miss classes from rankings.

    Hyperparameters
    ---------------
    strategy : {"random", "top"}, default="random"
        The strategy used to miss the classes from the rankings.
        Supported criteria are "random", to miss classes at
        random and "top", to miss the classes out of the top-k.

    percentage : float, default=0.0
        The percentage of classes to be missed.

    random_state : int or RandomState instance, default=None
        If ``int``, ``random_state`` is the seed used by the
        random number generator. If ``RandomState`` instance,
        ``random_state`` is the random number generator.
        If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    Attributes
    ----------
    n_samples_ : int
        The number of samples.

    n_classes_ : int
        The number of classes.

    n_missed_classes_ : int
        The number of classes to be missed.

    target_types_ : list of str
        The type of targets of the rankings.

    random_state_: RandomState instance
        ``RandomState`` instance that is generated either from
        a seed, the random number generator or by ``np.random``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.miss import SimpleMisser
    >>> Y = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    >>> misser = SimpleMisser("random", percentage=0.6, random_state=0)
    >>> misser.fit_transform(Y)
    array([[nan,  1.,  2.],
           [nan, nan,  1.],
           [nan,  1.,  2.]])
    >>> misser = SimpleMisser("top", percentage=0.6, random_state=0)
    >>> misser.fit_transform(Y)
    array([[ 1.,  2., inf],
           [ 2., inf,  1.],
           [inf,  1.,  2.]])
    """

    def __init__(self, strategy="random", percentage=0.0, random_state=None):
        """Constructor."""
        # Initialize the hyperparameters
        self.strategy = strategy
        self.percentage = percentage
        self.random_state = random_state

    def fit(self, Y):
        """Fit the misser on ``Y``.

        Parameters
        ----------
        Y : np.ndarray of shape (n_samples, n_classes)
            The input rankings.

        Returns
        -------
        self : SimpleMisser
            The fitted misser.

        Raises
        ------
        TypeError
            If the percentage of classes to be
            missed is not integer or floating.

        ValueError
            If the percentage of classes to be missed is not
            greater or equal than zero and less or equal than one.
        """
        # Validate the training rankings
        Y = self._validate_training_rankings(Y)

        # Check that the strategy takes a valid value
        if self.strategy not in {"random", "top"}:
            raise ValueError("The strategy must be either "
                             "'random' or 'top'. Got '{}'."
                             .format(self.strategy))

        # Check that the percentage of classes to be missed is integer
        # or floating (to ensure that undesired errors do not appear)
        if (not isinstance(self.percentage, (Integral, np.integer)) and
                not isinstance(self.percentage, (Real, np.floating))):
            raise TypeError("The percentage of classes to be missed "
                            "must be an integer or a floating type. "
                            "Got {}.".format(type(self.percentage)))

        # Check that the percentage of classes to be missed is
        # greater or equal than zero and less or equal than one
        if self.percentage < 0.0 or self.percentage > 1.0:
            raise ValueError("The percentage of classes to be missed must "
                             "be greater or equal than zero and less or "
                             "equal than one. Got {}."
                             .format(self.percentage))

        # Obtain the number of classes to be missed from each
        # input ranking, which is needed by the top-k strategy
        self.n_missed_classes_ = int(self.n_classes_ * self.percentage)

        # Obtain the random state instance, which is needed by the random
        # strategy to allow that the missed classes from rankings are seeded
        self.random_state_ = check_random_state(self.random_state)

        # Return the fitted misser
        return self

    def transform(self, Y):
        """Transform the rankings in ``Y``.

        Parameters
        ----------
        Y : np.ndarray of shape (n_samples, n_classes)
            The input rankings from which
            the classes will be missed.

        Returns
        -------
        Yt : np.ndarray of shape (n_samples, n_classes)
            The input rankings with missed classes
            according with the specified percentage.

        Raises
        ------
        ValueError
            If the type of targets of the input rankings
            is not a subset of the fitted type of targets.

        ValueError
            If the number of classes of the input rankings
            is different than the fitted number of classes.
        """
        # Check that the estimator is fitted
        check_is_fitted(self)

        # Validate the test data
        Y = self._validate_test_rankings(Y)

        # Initialize the number of samples and the
        # number of classes from the input rankings
        (n_samples, n_classes) = Y.shape

        # Initialize the rankings with missed classes
        # to afloating type (to allow infinite values)
        Yt = np.array(Y, dtype=np.float64)

        # Initialize an auxiliary set of
        # rankings to properly rank the data
        # when using the fast version of Cython
        Y = np.zeros(Y.shape, dtype=np.int64)

        # Obtain the masks to miss the classes from the
        # rankings according to the selected strategy
        if self.strategy == "random":
            # For the random strategy, miss the classes with probability
            # of being missed strictly less than the specified threshold
            masks = (self.random_state_.rand(n_samples, n_classes) <
                     self.percentage) | np.isnan(Yt)
        else:
            # For the top-k strategy, miss the classes that are
            # outside of top-k (in fact, argsort the rankings to
            # properly miss the classes when some of them are tied)
            masks = (np.argsort(np.argsort(Yt)) + 1 >
                     self.n_classes_ - self.n_missed_classes_)

        # Change the data type of the masks to ensure that
        # they can be managed by Cython extension module
        # in charge of missing the classes from the rankings
        masks = np.array(masks, dtype=np.uint8)

        # Obtain the strategy used by the Cython extension
        # module to miss the classes from the rankings
        strategy = STRATEGIES[self.strategy]

        # Miss the classes using the
        # optimized version of Cython
        miss_classes(Y, Yt, masks, strategy)

        # Return the transformed rankings
        return Yt
