"""Base classes for all transformers to miss classes from rankings."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from numbers import Integral, Real

# Third party
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

# Local application
from ..base import TransformerMixin
from ._base_fast import miss_classes
from ._base_fast import STRATEGY_MAPPING
from ..utils import check_random_state


# =============================================================================
# Classes
# =============================================================================

class SimpleMisser(TransformerMixin, BaseEstimator):
    """Missing transformer for deleting classes from rankings.

    Hyperparameters
    ---------------
    strategy : {"random", "top"}, default="random"
        The missing strategy. Supported criteria are ``"random"``, to delete
        the classes at random and ``"top"``, to delete the classes out of the
        top-k.

    probability : float, default=0.0
        The probability for the deletion of a class.

    random_state : int or RandomState instance, default=None
        Controls the randomness for the deletion of the classes.
        This parameter is useful only for the ``"random"`` strategy.

    Attributes
    ----------
    random_state_ : RandomState instance
        The random number generator to delete the classes.
        This attribute is available only for the ``"random"`` strategy.

    n_classes_miss_ : int
        The number of classes to delete from each ranking.
        This attribute is available only for the ``"top"`` strategy.

    Notes
    -----
    The values codifying the randomly and top-k deleted classes are ``np.nan``
    and ``np.inf`` (respectively) to maintain their underlying meaning.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.miss import SimpleMisser
    >>> Y = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    >>> misser = SimpleMisser("random", probability=0.6, random_state=0)
    >>> misser.fit_transform(Y)
    array([[nan,  1.,  2.],
           [nan, nan,  1.],
           [nan,  1.,  2.]])
    >>> misser = SimpleMisser("top", probability=0.6)
    >>> misser.fit_transform(Y)
    array([[ 1.,  2., inf],
           [ 2., inf,  1.],
           [inf,  1.,  2.]])
    """

    def __init__(self, strategy="random", probability=0.0, random_state=None):
        """Constructor."""
        self.strategy = strategy
        self.probability = probability
        self.random_state = random_state

    def fit(self, Y):
        """Fit the misser on ``Y``.

        Parameters
        ----------
        Y : ndarray of shape (n_samples, n_classes), dtype=np.int64
            The input rankings.

        Returns
        -------
        self : SimpleMisser
            The fitted misser.
        """
        Y = check_array(Y)

        (_, n_classes) = Y.shape

        if self.strategy not in STRATEGY_MAPPING:
            raise ValueError("The supported strategies are: {0}. Got '{1}'."
                             .format(list(STRATEGY_MAPPING), self.strategy))

        if (not isinstance(self.probability, (Integral, np.integer)) and
                not isinstance(self.probability, (Real, np.floating))):
            raise TypeError("The probability for the deletion of a "
                            "class must be integer or floating type. "
                            "Got {0}.".format(type(self.probability)))
        elif self.probability < 0.0 or self.probability > 1.0:
            raise ValueError("The probability for the deletion of a "
                             "class must be greater than or equal "
                             "zero and less than or equal one. "
                             "Got {0}.".format(self.probability))

        if self.strategy == "random":
            # Store the random number generator to guarantee
            # the reproducibility between successive fit calls
            self.random_state_ = check_random_state(self.random_state)
        else:
            self.n_classes_miss_ = int(n_classes * self.probability)

        return self

    def transform(self, Y):
        """Miss the classes in ``Y``.

        Parameters
        ----------
        Y : ndarray of shape (n_samples, n_classes), dtype=np.int64
            The input rankings from which the classes will be deleted.

        Returns
        -------
        Yt : ndarray of shape (n_samples, n_classes), dtype=np.float64
            The transformed rankings.
        """
        check_is_fitted(self)

        Y = check_array(Y)

        Yt = np.array(Y, dtype=np.float64)
        Y = np.zeros(Y.shape, dtype=np.int64)

        if self.strategy == "random":
            # Use an uniform distribution to randomly delete the classes
            masks = self.random_state_.rand(*Y.shape) < self.probability
        else:
            # Sort the rankings to mantain the arbitrariness for tied classes
            masks = np.argsort(np.argsort(-Yt)) + 1 <= self.n_classes_miss_

        # Transform the masks to unsigned integer
        # since Cython cannot handle boolean arrays
        masks = np.array(masks, dtype=np.uint8)
        strategy = STRATEGY_MAPPING[self.strategy]

        miss_classes(Y, Yt, masks, strategy)

        return Yt
