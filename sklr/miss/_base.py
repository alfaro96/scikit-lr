"""This module gathers base transformers to miss classes in rankings."""


# =============================================================================
# Imports
# =============================================================================

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
    """Misser transformer for missing classes in rankings.

    Hyperparameters
    ---------------
    strategy : str, optional (default="random")
        The strategy used to miss the classes from the rankings.
        Supported criteria are "random" for randomly missed classes
        and "top" to miss the classes outside of the "top-k".

    percentage : float, optional (default=0.0)
        The percentage of classes to miss.

    random_state : {int, RandomState instance, None}, optional (default=None)
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance
          used by `np.random`.

        Only employed if ``strategy="random"``.

    Attributes
    ----------
    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    n_missed_classes_ : int
        The number of classes to be missed (only available
        if ``strategy="top"``).

    target_types_ : dict of str
        The target types of the rankings provided at ``fit``.

    random_state_: np.random.RandomState instance
        The random state generator (only available
        if ``strategy="random"``).

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
        """Fit the transformer on Y.

        Parameters
        ----------
        Y : np.ndarray of shape (n_samples, n_classes)
            Input rankings, where ``n_samples`` is the number of samples
            and ``n_classes`` is the number of classes.

        Returns
        -------
        self : SimpleMisser

        Raises
        ------
        ValueError
            If the percentage of classes to be missed is not a value
            greater or equal than zero and less or equal than one.
        """
        # Validate the training rankings
        Y = self._validate_training_rankings(Y)

        # Check that the strategy is valid
        if self.strategy not in {"random", "top"}:
            raise ValueError("The strategy must be either "
                             "'random' or 'top'. Got '{}'."
                             .format(self.strategy))

        # Check that the percentage is value greater or
        # equal than zero and less or equal than one
        if self.percentage < 0.0 or self.percentage > 1.0:
            raise ValueError("The percentage of missed classes must be "
                             "a floating value greater or equal than zero "
                             "and less or equal than one. Got {}."
                             .format(self.percentage))

        # Obtain the random state, since it is needed
        # to miss the rankings in the transform method
        # when the selected strategy is "random"
        if self.strategy == "random":
            self.random_state_ = check_random_state(self.random_state)
        # Otherwise, obtain the number of classes to be
        # miss on each ranking for the "top" strategy
        else:
            self.n_missed_classes_ = int(self.n_classes_
                                         * self.percentage)

        # Return the fitted transformer
        return self

    def transform(self, Y):
        """Miss the classes in Y.

        Parameters
        ----------
        Y : np.ndarray of shape (n_samples, n_classes)
            The input rankings to miss.

        Returns
        -------
        Yt : np.ndarray of shape (n_samples, n_classes)
            The missed input rankings.

        Raises
        ------
        ValueError
            If the target types of the input rankings is not a subset
            of the fitted target types.

        ValueError
            If the number of classes of the input rankings is
            different than the fitted number of classes.
        """
        # Check if the estimator is fitted
        check_is_fitted(self)

        # Validate the test data
        Y = self._validate_test_rankings(Y)

        # Initialize the number of samples and the
        # number of classes from the input rankings
        (n_samples, n_classes) = Y.shape

        # Initialize the missed rankings, using
        # a floating type to allow infinite values
        Yt = np.array(Y, dtype=np.float64)

        # Initialize an auxiliary set of rankings to properly
        # rank the data when using the fast version of Cython
        Y = np.zeros(Y.shape, dtype=np.int64)

        # Obtain the masks
        if self.strategy == "random":
            # For the random strategy, miss the classes
            # that have a probability of being missed
            # less than the specified threshold
            masks = (self.random_state_.rand(n_samples, n_classes) <
                     self.percentage) | np.isnan(Yt)
        else:
            # For the top-k strategy, miss the classes that are outside
            # of top-k (in fact, argsort the rankings to properly
            # miss the classes when some of them are tied)
            masks = np.argsort(np.argsort(Yt)) + 1 > (self.n_classes_
                                                      - self.n_missed_classes_)

        # Change the data type of the masks to ensure that
        # they can be managed by Cython extension module
        # in charge of missing the classes from the rankings
        masks = np.array(masks, dtype=np.uint8)

        # Miss the classes using the
        # optimized version of Cython
        miss_classes(
            Y, Yt, masks,
            strategy=STRATEGIES[self.strategy])

        # Return the transformed rankings
        return Yt
