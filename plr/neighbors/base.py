"""
    Base classes for nearest-neighbors estimators.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# Scipy
from scipy.spatial          import KDTree
from scipy.spatial.distance import minkowski

# PLR
from ..base             import BasePartialLabelRanker
from ..utils.validation import check_random_state, check_X_Y_sample_weight

# Misc
from abc import abstractmethod

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
                 bucket,
                 beta,                 
                 p,
                 leaf_size,
                 random_state):
        """
            Base constructor for the nearest-neighbors paradigm.
        """
        # Call to the constructor of the parent
        super().__init__(bucket       = bucket,
                         beta         = beta,
                         random_state = random_state)

        # Initialize other hyperparameters of the current object
        self.algorithm   = algorithm
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

                sample_weight: {None, np.ndarray} (default = None)
                    The sample weight of each instance. If "None", the samples are equally weighted.

                check_input: boolean (default = True)
                    Allow to bypass several input checking.

            Returns
            -------
                self:
                    Current object already trained.
        """
        # Check the input parameters (if corresponds)
        if check_input:
            (X, Y, sample_weight) = check_X_Y_sample_weight(X, Y, sample_weight)

        # Initialize some attributes of the current object
        (self.n_samples_, self.n_features_)     = X.shape
        self.n_classes_                         = Y.shape[1]
        (self.X_, self.Y_, self.sample_weight_) = (X, Y, sample_weight)

        # Initialize the random state
        self.random_state_ = check_random_state(self.random_state)

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
                 n_neighbors  = 5,
                 algorithm    = "kd_tree",
                 bucket       = "bpa_lia_mp2",
                 beta         = 0.25,
                 p            = 2,
                 leaf_size    = 30,
                 random_state = None):
        """
            Constructor of BaseKNearestNeighborsPartialLabelRanker class.
            
            Parameters
            ----------
                n_neighbors: int, optional (default = 5)
                    Number of neighbors to use.
                       
                algorithm: string, optional (default = "kd_tree")
                    Algorithm used to compute the nearest neighbors:
                        - "brute" will use a brute-force search.
                        - "kd_tree" will use a KD-Tree.

                bucket: string, optional (default = "bpa_lia_mp2")
                    Algorithm that will be applied to the OBOP problem, i.e., the algorithm employed
                    to aggregate the bucket orders.

                beta: float, optional (default = 0.25)
                    The parameter to decide the precedence relation of each item w.r.t. the pivot.
            
                p: int, optional (default = 2)
                    Power parameter for the Minkowski metric.
                    When p = 1, this is equivalent to using Manhattan distance,
                    and Euclidean distance for p = 2. 
                    For arbitrary p, Minkowski distance is used.

                leaf_size: int, optional (default = 30)
                    Leaf size passed to KDTree. This can affect the speed of the construction and query,
                    as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

                random_state: {None, int, RandomState instance}, optional (default = None)
                    - If int, random_state is the seed used by the random number generator.
                    - If RandomState instance, random_state is the random number generator.
                    - If None, the random number generator is the RandomState instance used
                      by np.random.
        
            Returns
            -------
                self: object
                    Current object initialized.
        """
        # Call to the constructor of the parent
        super().__init__(algorithm    = algorithm,
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
            neighbors = np.array([np.argsort(np.array([minkowski(X[i],
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
