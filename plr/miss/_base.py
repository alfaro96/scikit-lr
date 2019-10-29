"""This module gathers base transformers to miss classes in rankings."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from math import ceil

# Third party
import numpy as np

# Local application
from ..base import BaseEstimator, TransformerMixin
from ..utils.ranking import type_of_targets, rank_data
from ..utils.validation import check_array, check_is_fitted, check_random_state


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
    percentage : float, optional (default=0.0)
        The percentage of classes to miss.

    strategy : str, optional (default="random")
        The miss strategy.

        - If "random", then randomly miss the classes.
        - If "top", then miss the classes out of the top-k.

    random_state: {int, RandomState instance, None}, optional (default=None)
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance
          used by `np.random`.

    Attributes
    ----------
    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    n_missed_classes_ : int
        The number of classes to miss. This attribute is only interesting
        when ``strategy="top"``.

    target_types_ : dict of str
        The target types of the rankings provided at ``fit``.

    random_state_: np.random.RandomState instance
        The random state generator. This attribute is only interesting
        when ``strategy="random"``.

    Examples
    --------
    >>> import numpy as np
    >>> from plr.miss import SimpleMisser
    >>> Y = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    >>> mis = SimpleMisser(percentage=0.6, strategy="random", random_state=0)
    >>> mis.fit_transform(Y)
    array([[nan,  1.,  2.],
           [nan, nan,  1.],
           [nan,  1.,  2.]])
    >>> mis = SimpleMisser(percentage=0.6, strategy="top")
    >>> mis.fit_transform(Y)
    array([[ 1., inf, inf],
           [inf, inf,  1.],
           [inf,  1., inf]])
    """

    def __init__(self, percentage=0.0, strategy="random", random_state=None):
        """Constructor."""
        # Initialize the hyperparameters
        self.percentage = percentage
        self.strategy = strategy
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
        """
        # Validate the training data
        Y = self._validate_training_data(Y)

        # Check that the strategy is correct
        if self.strategy not in {"random", "top"}:
            raise ValueError("The supported criteria are \"random\" "
                             "and \"top\". Got \"{}\"."
                             .format(self.strategy))

        # Check that the percentage is correct
        if self.percentage < 0.0 or self.percentage > 1.0:
            raise ValueError("The percentage of missed classes must be "
                             "a floating value greater or equal than zero "
                             "and less or equal than one. Got {}."
                             .format(self.percentage))

        # Obtain the number of classes to miss
        self.n_missed_classes_ = ceil(self.n_classes_ * self.percentage)

        # Obtain the random state
        self.random_state_ = check_random_state(self.random_state)

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
        Y = self._validate_test_data(Y)

        # Obtain the number of samples and the number of classes
        (n_samples, n_classes) = Y.shape

        # Check if the rankings already contain missed classes
        if np.any(~np.isfinite(Y)):
            raise ValueError("The rankings already contain missed classes.")

        # Initialize the missed rankings
        Yt = np.array(Y, dtype=np.float64)

        # Obtain the masks (according to the selected strategy)
        if self.strategy == "random":
            masks = (self.random_state_.rand(n_samples, n_classes) <
                     self.percentage)
        else:
            n_keep_classes = (self.n_classes_ - self.n_missed_classes_)
            masks = np.argsort(Y, axis=1)[:, n_keep_classes:]

        # Miss the rankings
        for sample in range(n_samples):
            # For efficiency purposes, only miss classes
            # on the rankings where any class must be missed
            if masks[sample].shape[0] != 0:
                # For the random strategy, rank the data
                # to mantain a properly formatted ranking
                if self.strategy == "random":
                    Yt[sample, ~masks[sample]] = rank_data(
                        data=Y[sample, ~masks[sample]],
                        method="dense")
                # Miss the classes from the ranking
                if self.strategy == "random":
                    Yt[sample, masks[sample]] = np.nan
                else:
                    Yt[sample, masks[sample]] = np.inf

        # Return the missed rankings
        return Yt
