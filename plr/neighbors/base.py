"""
    Base classes for nearest-neighbors estimators.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# Scipy
from scipy.spatial import KDTree

# PLR
from ..base                import BasePartialLabelRanker
from ..bucket.obop         import ALGORITHMS as OBOP_ALGORITHMS
from ._builder             import KNeighborsBuilder
from ..metrics._calculator import minkowski_calculator
from ..utils.validation    import check_random_state, check_X_Y_sample_weight

# Misc
from abc import abstractmethod

# =============================================================================
# Public objects
# =============================================================================

# Builders
BUILDERS = {
                "knn": KNeighborsBuilder
           }

# =============================================================================
# Base nearest neighbors
# =============================================================================
class BaseNeighborsPartialLabelRanker(BasePartialLabelRanker):
    """
        Base class for the nearest neighbors paradigm.
    """

    @abstractmethod    
    def __init__(self,
                 algorithm,
                 builder,
                 bucket,
                 beta,                 
                 p,
                 leaf_size,
                 random_state):
        """
            Base constructor for the BaseNeighborsPartialLabelRanker class.
        """
        # Call to the constructor of the parent
        super().__init__(bucket       = bucket,
                         beta         = beta,
                         random_state = random_state)

        # Initialize other hyperparameters of the current object
        self.algorithm   = algorithm
        self.builder     = builder
        self.p           = p
        self.leaf_size   = leaf_size

    def fit(self,
            X,
            Y,
            sample_weight = None,
            check_input   = True):
        """
            Fit the corresponding nearest-neighbors model.

            Parameters
            ----------
                X: np.ndarray
                    The training input samples.

                Y: np.ndarray
                    The target values (bucket orders) as array.

                sample_weight: {None, np.ndarray}, optional (default = None)
                    Weights already sorted by distance.
                
                check_input: boolean (default = True)
                    Allow to bypass several input checking.

            Returns
            -------
                self: BaseNeighborsPartialLabelRanker
                    Current object already trained.
        """
        # Check the input parameters (if corresponds)
        if check_input:
            (X, Y, sample_weight) = check_X_Y_sample_weight(X, Y, sample_weight)

        # Initialize some attributes of the current object
        (self.n_samples_, self.n_features_)     = X.shape
        self.n_classes_                         = Y.shape[1]
        (self.X_, self.Y_, self.sample_weight_) = (X, Y, sample_weight)

        # Obtain the random state
        random_state = check_random_state(self.random_state)

        # Initialize the builder for the current object
        self.builder_ = BUILDERS[self.builder](bucket = OBOP_ALGORITHMS[self.bucket], beta = self.beta, random_state = random_state).init(self.n_classes_)

        # Obtain the data structure to store the training instances
        # Brute-force algorithm has not underlyling structure
        if self.algorithm == "brute":
            self.tree_ = None
        # KD-Tree has the one obtained from scipy
        elif self.algorithm == "kd_tree":
            self.tree_ = KDTree(data     = self.X_,
                                leafsize = self.leaf_size)
        # Otherwise, the algorithm is not implemented
        else:
            raise NotImplementedError("The algorithm '{}' is not implemented. Check the input parameters.".format(self.algorithm))
                        
        # Return the current object already trained
        return self

# =============================================================================
# Base k-nearest neighbors
# =============================================================================
class BaseKNeighborsPartialLabelRanker(BaseNeighborsPartialLabelRanker):
    """
        Base class for the k-nearest neighbors paradigm.
    """
    
    def __init__(self,       
                 n_neighbors,
                 algorithm,
                 builder,
                 bucket,
                 beta,
                 p,
                 leaf_size,
                 random_state):
        """
            Base constructor of BaseKNearestNeighborsPartialLabelRanker class.
        """
        # Call to the constructor of the parent
        super().__init__(algorithm    = algorithm,
                         builder      = builder,
                         bucket       = bucket,
                         beta         = beta,
                         p            = p,
                         leaf_size    = leaf_size,
                         random_state = random_state)

        # Initialize other hyperparameters of the current object
        self.n_neighbors = n_neighbors

    def knearestneighbors(self,
                          X):
        """
            Obtain the k-nearest neighbors of the test dataset regarding the training one.

            Parameters
            ----------
                X: np.ndarray
                    The test input samples.

            Returns
            -------
                neighbors: np.ndarray
                    Indexes of the k-nearest neighbors.
        """
        # Initialize some values from the input arrays
        n_samples = X.shape[0]

        # Obtain the nearest neighbors using the corresponding algorithm
        # If it is None, use the brute-force search
        if isinstance(self.tree_, type(None)):
            neighbors = np.array([np.argsort(np.array([minkowski_calculator(X[i],
                                                                            self.X_[j],
                                                                            self.p)
                                                       for j in range(self.n_samples_)]))[:self.n_neighbors]
                                  for i in range(n_samples)])
        # Otherwise, use the KD-Tree
        else:
            neighbors = self.tree_.query(x = X,
                                         k = min(self.n_neighbors, self.n_samples_), # Avoid out of bounds errors when self.n_neighbors > self.n_samples_
                                         p = self.p)[1]

        # Return the obtained indexes
        return neighbors
