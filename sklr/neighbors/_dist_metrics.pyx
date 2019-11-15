# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Standard
from numbers import Integral, Real

# Third party
from libc.math cimport sqrt, fabs, fmax
from libc.math cimport INFINITY
import numpy as np
cimport numpy as np

# Always include this statement after cimporting
# NumPy, to avoid segmentation faults
np.import_array()

# Local application
from ..utils.validation import check_array


# =============================================================================
# Public objects
# =============================================================================

# The metric mapping to each distance
# function that can be employed
METRIC_MAPPING = {
    "manhattan": ManhattanDistance,
    "cityblock": ManhattanDistance,
    "l1": ManhattanDistance,
    "euclidean": EuclideanDistance,
    "l2": EuclideanDistance,
    "chebyshev": ChebyshevDistance,
    "infinity": ChebyshevDistance,
    "minkowski": MinkowskiDistance,
    "p": MinkowskiDistance
}


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Distance metric
# =============================================================================
cdef class DistanceMetric:
    """DistanceMetric class.

    This class provides a uniform interface to fast distance metric
    functions. The various metrics can be accessed via the :meth:`get_metric`
    class method and the metric string identifier (see below).
    For example, to use the Euclidean distance:

    >>> import numpy as np
    >>> from sklr.neighbors import DistanceMetric
    >>> dist = DistanceMetric.get_metric("euclidean")
    >>> X = np.array([[0, 1, 2], [3, 4, 5]])
    >>> dist.pairwise(X)
    array([[ 0.        ,  5.19615242],
           [ 5.19615242,  0.        ]])

    Available Metrics

    The following lists the string metric identifiers and the associated
    distance metric classes:

    ===========  =================  ====  ========================
    identifier   class name         args  distance function
    -----------  -----------------  ----  ------------------------
    "euclidean"  EuclideanDistance  -     ``sqrt(sum((x - y)^2))``
    "manhattan"  ManhattanDistance  -     ``sum(|x - y|)``
    "chebyshev"  ChebyshevDistance  -     ``max(|x - y|)``
    "minkowski"  MinkowskiDistance  p     ``sum(|x - y|^p)^(1/p)``
    ===========  =================  ====  ========================
    """

    def __cinit__(self):
        """Constructor."""
        # Initialize the hyperparameters
        # to some default values
        self.p = 2

    def __init__(self):
        """Constructor."""
        if self.__class__ is DistanceMetric:
            raise NotImplementedError("DistanceMetric is an abstract class.")

    @classmethod
    def get_metric(cls, metric, **kwargs):
        """Get the given distance metric from the string identifier.

        Parameters
        ----------
        metric : str
            The distance metric to use.

        **kwargs : dict
            Additional arguments will be passed to the requested metric.

        Returns
        -------
        dist_metric : DistanceMetric
            The distance metric.

        Raises
        ------
        ValueError
            If the provided metric is not available.
        """
        # Map the metric string ID to the distance metric class
        if (isinstance(metric, type) and
                issubclass(metric, DistanceMetric)):
            pass
        else:
            try:
                dist_metric = METRIC_MAPPING[metric]
            except KeyError:
                raise ValueError("Unrecognized metric: '{}'. "
                                 "Valid metrics are: {}."
                                 .format(
                                     metric,
                                     sorted(list(METRIC_MAPPING.keys()))))

        # Obtain the power parameter
        # for the Minkowski distance
        # from the keyworked arguments
        # (default value of 2)
        p = kwargs.pop("p", 2)

        # In Minkowski special cases, return
        # more efficient distance methods
        if dist_metric is MinkowskiDistance:
            if p == 1:
                dist_metric = ManhattanDistance(**kwargs)
            elif p == 2:
                dist_metric = EuclideanDistance(**kwargs)
            elif p == INFINITY:
                dist_metric = ChebyshevDistance(**kwargs)
            else:
                dist_metric = MinkowskiDistance(p, **kwargs)
        else:
            dist_metric = dist_metric(**kwargs)

        # Return the distance metric
        return dist_metric

    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil:
        """Compute the distance between vectors x1 and x2.

        This should be overridden in a subclass."""
        pass

    cdef void pdist(self, DTYPE_t_2D X, DTYPE_t_2D dist) nogil:
        """Compute the pairwise distances between points in X."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = X.shape[0]

        # Define the indexes
        cdef SIZE_t f_sample
        cdef SIZE_t s_sample

        # Compute the pairwise distances
        # between all pair of samples
        for f_sample in range(n_samples):
            for s_sample in range(f_sample, n_samples):
                dist[f_sample, s_sample] = self.dist(X[f_sample], X[s_sample])
                dist[s_sample, f_sample] = dist[f_sample, s_sample]

    cdef void cdist(self, DTYPE_t_2D X1, DTYPE_t_2D X2, DTYPE_t_2D dist) nogil:
        """Compute the cross-pairwise distances between X1 and X2."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples_X1 = X1.shape[0]
        cdef INT64_t n_samples_X2 = X2.shape[0]

        # Define the indexes
        cdef SIZE_t f_sample
        cdef SIZE_t s_sample

        # Compute the cross-pairwise distances
        # between the given input samples
        for f_sample in range(n_samples_X1):
            for s_sample in range(n_samples_X2):
                dist[f_sample, s_sample] = self.dist(
                    X1[f_sample], X2[s_sample])

    def pairwise(self, X1, X2=None):
        """Compute the pairwise distances between X1 and X2.

        This is a convenience routine for the sake of testing. For many
        metrics, the utilities in scipy.spatial.distance.cdist and
        scipy.spatial.distance.pdist will be faster.

        Parameters
        ----------
        X1 : np.ndarray of shape (n_samples, n_features)
            Array representing Nx1 points in D dimensions.

        X2 : np.ndarray of shape (n_samples, n_features)
            Array representing Nx2 points in D dimensions.
            If not specified, then X1=X2.

        Returns
        -------
        dist : np.ndarray of shape (n_samples, n_samples)
            The shape (Nx1, Nx2) array of pairwise distances
            between points in X1 and X2.
        """
        # Check the format of the input array forcing it
        # to be a floating type for Cython typing purposes
        check_array(X1, dtype=np.float64)

        # Define some values to be employed
        cdef np.ndarray[DTYPE_t, ndim=2] dist

        # If X2 is not provided, compute the pairwise
        # distances between the points in X1
        if X2 is None:
            # Initialize the array of distances
            dist = np.zeros((X1.shape[0], X1.shape[0]), dtype=np.float64)
            # Compute the pairwise distances
            self.pdist(X1, dist)
        # Otherwise, compute the cross-pairwise
        # distances between arrays X1 and X2
        else:
            # Check the format of the input array forcing it
            # to be a floating type for Cython typing purposes
            X2 = check_array(X2, dtype=np.float64)
            # Ensure that X1 and X2 have
            # the same second dimension
            if X1.shape[1] != X2.shape[1]:
                raise ValueError("X1 and X2 must have the same second "
                                 "dimension. Got X1 = {} and X2 = {}."
                                 .format(X1.shape, X2.shape))
            # Initialize the array of distances
            dist = np.zeros((X1.shape[0], X2.shape[0]), dtype=np.float64)
            # Compute the cross parwise distances
            self.cdist(X1, X2, dist)

        # Return the pairwise distances
        return dist


# =============================================================================
# Manhattan distance
# =============================================================================
cdef class ManhattanDistance(DistanceMetric):
    """Manhattan distance metric."""

    def __init__(self):
        """Constructor."""
        # Initialize the hyperparameters
        self.p = 1

    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil:
        """Compute the Manhattan distance between vectors x1 and x2."""
        # Define some values to be employed
        cdef INT64_t n_features = x1.shape[0]

        # Define some values to be employed
        cdef DTYPE_t man_dist

        # Define the indexes
        cdef SIZE_t att

        # Initialize the Manhattan distance to zero
        man_dist = 0.0

        # Compute the Manhattan distance
        for att in range(n_features):
            man_dist += fabs(x1[att] - x2[att])

        # Return it
        return man_dist


# =============================================================================
# Euclidean distance
# =============================================================================
cdef class EuclideanDistance(DistanceMetric):
    """Euclidean distance metric."""

    def __init__(self):
        """Constructor."""
        # Initialize the hyperparameters
        self.p = 2

    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil:
        """Compute the Euclidean distance between vectors x1 and x2."""
        # Define some values to be employed
        cdef INT64_t n_features = x1.shape[0]

        # Define some values to be employed
        cdef DTYPE_t euc_dist

        # Define the indexes
        cdef SIZE_t att

        # Initialize the Euclidean distance to zero
        euc_dist = 0.0

        # Compute the Euclidean distance
        for att in range(n_features):
            euc_dist += (x1[att]-x2[att]) ** 2
        euc_dist = sqrt(euc_dist)

        # Return it
        return euc_dist


# =============================================================================
# Chebyshev distance
# =============================================================================
cdef class ChebyshevDistance(DistanceMetric):
    """Chebyshev distance metric."""

    def __init__(self):
        """Constructor."""
        # Initialize the hyperparameters
        self.p = INFINITY

    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil:
        """Compute the Chebyshev distance between vectors x1 and x2."""
        # Define some values to be employed
        cdef INT64_t n_features = x1.shape[0]

        # Define some values to be employed
        cdef DTYPE_t che_dist

        # Define the indexes
        cdef SIZE_t att

        # Initialize the Chebyshev distance to zero
        che_dist = 0.0

        # Compute the Chebyshev distance
        for att in range(n_features):
            che_dist = fmax(che_dist, fabs(x1[att] - x2[att]))

        # Return it
        return che_dist


# =============================================================================
# Minkowski distance
# =============================================================================
cdef class MinkowskiDistance(DistanceMetric):
    """Minkowski distance metric."""

    def __init__(self, p):
        """Constructor."""
        # Check the power hyperparameter, ensuring that
        # is a floating or integer type taking values
        # greater than or equal one and less than infinity
        if (not isinstance(p, (Real, np.floating))
                and not isinstance(p, (Integral, np.integer))):
            raise TypeError("Expected p to take a float or int value. Got {}."
                            .format(type(p).__name__))
        elif p < 1:
            raise ValueError("p must be greater than 1.".format(p))
        elif p == INFINITY:
            raise ValueError("MinkowskiDistance requires finite p. "
                             "For p=inf, use ChebyshevDistance.")

        # Initialize the hyperparameters
        self.p = p

    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil:
        """Compute the Minkowski distance between vectors x1 and x2."""
        # Define some values to be employed
        cdef INT64_t n_features = x1.shape[0]

        # Define some values to be employed
        cdef DTYPE_t min_dist

        # Define the indexes
        cdef SIZE_t att

        # Initialize the Minkowski distance to zero
        min_dist = 0.0

        # Compute the Minkowski distance
        for att in range(n_features):
            min_dist += fabs(x1[att]-x2[att]) ** self.p
        min_dist = min_dist ** (1.0/self.p)

        # Return it
        return min_dist
