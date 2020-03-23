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
# NumPy, to avoid possible segmentation faults
np.import_array()

# Local application
from ..utils.validation import check_array


# =============================================================================
# Public objects
# =============================================================================

# Map the metric identifier to the class name
METRIC_MAPPING = {
    "manhattan": ManhattanDistance,
    "euclidean": EuclideanDistance,
    "minkowski": MinkowskiDistance,
    "chebyshev": ChebyshevDistance
}


# =============================================================================
# Classes
# =============================================================================

cdef class DistanceMetric:
    """A distance metric wrapper.

    This class provides a uniform interface to fast distance metric functions.
    The various metrics can be accessed via the :meth:`get_metric` class
    method and the metric string identifier (see below).

    ===========  =================  ====  ========================
    identifier   class name         args  distance function
    -----------  -----------------  ----  ------------------------
    "euclidean"  EuclideanDistance  -     ``sqrt(sum((x - y)^2))``
    "manhattan"  ManhattanDistance  -     ``sum(|x - y|)``
    "chebyshev"  ChebyshevDistance  -     ``max(|x - y|)``
    "minkowski"  MinkowskiDistance  p     ``sum(|x - y|^p)^(1/p)``
    ===========  =================  ====  ========================

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.neighbors import DistanceMetric
    >>> dist = DistanceMetric.get_metric("euclidean")
    >>> X = np.array([[0, 1, 2], [3, 4, 5]])
    >>> dist.pairwise(X)
    array([[ 0.        ,  5.19615242],
           [ 5.19615242,  0.        ]])
    """

    def __cinit__(self):
        """Constructor."""
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
        metric : {"manhattan", "euclidean", "minkowski", "chebyshev"}
            The distance metric to use.

        **kwargs : dict
            The additional arguments that will be passed to the requested
            metric.

        Returns
        -------
        dist_metric : DistanceMetric
            The distance metric.
        """
        # Re-raise a more informative message when the metric is not valid
        try:
            dist_metric = METRIC_MAPPING[metric]
        except KeyError:
            raise ValueError("Unknown metric: {0}. Expected one of {1}."
                             .format(metric, list(METRIC_MAPPING)))

        p = kwargs.pop("p", 2)

        # For special cases of the Minkowski distance function,
        # it is possible to return more efficient distance methods
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

        return dist_metric

    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil:
        """Compute the distance between vectors x1 and x2.

        This should be overridden in a subclass.
        """
        pass

    cdef void pdist(self, DTYPE_t_2D X, DTYPE_t_2D dist) nogil:
        """Compute the pairwise distances between points in X."""
        cdef INT64_t n_samples = X.shape[0]

        cdef SIZE_t f_sample
        cdef SIZE_t s_sample

        for f_sample in range(n_samples):
            for s_sample in range(f_sample, n_samples):
                dist[f_sample, s_sample] = self.dist(X[f_sample], X[s_sample])
                dist[s_sample, f_sample] = dist[f_sample, s_sample]

    cdef void cdist(self, DTYPE_t_2D X1, DTYPE_t_2D X2, DTYPE_t_2D dist) nogil:
        """Compute the cross-pairwise distances between X1 and X2."""
        cdef INT64_t n_samples_X1 = X1.shape[0]
        cdef INT64_t n_samples_X2 = X2.shape[0]

        cdef SIZE_t f_sample
        cdef SIZE_t s_sample

        for f_sample in range(n_samples_X1):
            for s_sample in range(n_samples_X2):
                dist[f_sample, s_sample] = self.dist(X1[f_sample],
                                                     X2[s_sample])

    def pairwise(self, X1, X2=None):
        """Compute the pairwise distances between ``X1`` and ``X2``.

        This is a convenience routine for the sake of testing. For many
        metrics, the utilities in ``scipy.spatial.distance.cdist`` and
        ``scipy.spatial.distance.pdist`` will be faster.

        Parameters
        ----------
        X1 : ndarray of shape (n_samples_X1, n_features), dtype=np.float64
            The array representing `Nx1` points in `D` dimensions.

        X2 : ndarray of shape (n_samples_X2, n_features), dtype=np.float64, \
                default=None
            The array representing `Nx2` points in `D` dimensions.
            If ``None``, then ``X1=X2``.

        Returns
        -------
        dist : ndarray of shape (n_samples_X1, n_samples_X2), dtype=np.float64
            The shape `(Nx1, Nx2)` array of pairwise distances between
            points in ``X1`` and ``X2``.
        """
        X1 = check_array(X1, dtype=np.float64)

        cdef np.ndarray[DTYPE_t, ndim=2] dist

        if X2 is None:
            # The pairwise distances is a (Nx1, Nx1) matrix because
            # the distance between the points in "X1" must be computed
            dist = np.zeros((X1.shape[0], X1.shape[0]), dtype=np.float64)
            self.pdist(X1, dist)
        else:
            # The pairwise distances is a (Nx1, Nx2) matrix because
            # the distance between the points in "X1" and "X2" must
            # be computed. Note that the points in "X1" are indexed
            # by row and the points in "X2" are indexed by column
            X2 = check_array(X2, dtype=np.float64)
            dist = np.zeros((X1.shape[0], X2.shape[0]), dtype=np.float64)
            self.cdist(X1, X2, dist)

        return dist


cdef class ManhattanDistance(DistanceMetric):
    """Manhattan distance metric."""

    def __init__(self):
        """Constructor."""
        self.p = 1

    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil:
        """Compute the Manhattan distance between vectors x1 and x2."""
        cdef INT64_t n_features = x1.shape[0]
        cdef DTYPE_t man_dist = 0.0

        cdef SIZE_t att

        for att in range(n_features):
            man_dist += fabs(x1[att] - x2[att])

        return man_dist


cdef class EuclideanDistance(DistanceMetric):
    """Euclidean distance metric."""

    def __init__(self):
        """Constructor."""
        self.p = 2

    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil:
        """Compute the Euclidean distance between vectors x1 and x2."""
        cdef INT64_t n_features = x1.shape[0]
        cdef DTYPE_t euc_dist = 0.0

        cdef SIZE_t att

        for att in range(n_features):
            euc_dist += (x1[att]-x2[att]) ** 2
        euc_dist = sqrt(euc_dist)

        return euc_dist


cdef class MinkowskiDistance(DistanceMetric):
    """Minkowski distance metric."""

    def __init__(self, p):
        """Constructor."""
        if (not isinstance(p, (Real, np.floating))
                and not isinstance(p, (Integral, np.integer))):
            raise TypeError("Expected power parameter to take a floating or "
                            "integer value. Got {0}.".format(type(p).__name__))
        elif p < 1:
            raise ValueError("The power parameter be greater "
                             "than one. Got {0}.".format(p))
        elif p == INFINITY:
            raise ValueError("Minkowski distance requires finite power "
                             "parameter. Use Chebyshev distance instead.")

        self.p = p

    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil:
        """Compute the Minkowski distance between vectors x1 and x2."""
        cdef INT64_t n_features = x1.shape[0]
        cdef DTYPE_t min_dist = 0.0

        cdef SIZE_t att

        for att in range(n_features):
            min_dist += fabs(x1[att]-x2[att]) ** self.p
        min_dist = min_dist ** (1.0/self.p)

        return min_dist


cdef class ChebyshevDistance(DistanceMetric):
    """Chebyshev distance metric."""

    def __init__(self):
        """Constructor."""
        self.p = INFINITY

    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil:
        """Compute the Chebyshev distance between vectors x1 and x2."""
        cdef INT64_t n_features = x1.shape[0]
        cdef DTYPE_t che_dist = 0.0

        cdef SIZE_t att

        for att in range(n_features):
            che_dist = fmax(che_dist, fabs(x1[att]-x2[att]))

        return che_dist
