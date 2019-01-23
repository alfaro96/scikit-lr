"""
    Base classes for decision tree estimators.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from ..base             import BasePartialLabelRanker
from ..bucket._builder  import normalize_sample_weight
from ..bucket.obop      import ALGORITHMS as OBOP_ALGORITHMS
from ._builder          import GreedyBuilder
from ._criterion        import EntropyCriterion, DisagreementsCriterion, DistanceCriterion
from ._splitter         import BinarySplitter
from ..utils.validation import check_is_fitted, check_n_features, check_random_state, check_X, check_X_Y_sample_weight

# Misc
from abc import ABC, abstractmethod

# =============================================================================
# Public objects
# =============================================================================

# Variables
MAX_INT = np.iinfo(np.int64).max

# Algorithms
BUILDERS = {
                "greedy": GreedyBuilder
           }

# Splitters
SPLITTERS = {
                "binary": BinarySplitter
            }

# Criteria
CRITERIA = {
                "disagreements": DisagreementsCriterion,
                "distance":      DistanceCriterion,
                "entropy":       EntropyCriterion
           }

CRITERIA_REQUIRING_OBOP = {"disagreements", "distance"}

# =============================================================================
# Base decision tree
# =============================================================================
class BaseDecisionTreePartialLabelRanker(BasePartialLabelRanker):
    """
        Base class for decision trees.
    """

    @abstractmethod
    def __init__(self,
                 algorithm,
                 criterion,
                 splitter,
                 bucket,
                 beta,
                 max_depth,
                 min_samples_split,
                 max_features,
                 max_splits,
                 random_state):
        # Call to the constructor of the parent
        super().__init__(bucket       = bucket,
                         beta         = beta,
                         random_state = random_state)

        # Initialize other hyperparameters of the current object
        self.algorithm         = algorithm
        self.criterion         = criterion
        self.splitter          = splitter
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.max_splits        = max_splits

    def fit(self,
            X,
            Y,
            sample_weight = None,
            check_input   = True):
        """
            Build a decision tree classifier from the training set (X, Y).

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
                self: BaseDecisionTreePartialLabelRanker
                    Current object already trained.
        """
        # Check the input parameters (if corresponds)
        if check_input:
            (X, Y, sample_weight) = check_X_Y_sample_weight(X, Y, sample_weight)

        # Initialize the attributes of the current object
        (self.n_samples_, self.n_features_) = X.shape
        self.n_classes_                     = Y.shape[1]
        
        # Obtain the random state
        random_state = check_random_state(self.random_state)

        # Infer the maximum depth
        if self.max_depth == None:
            max_depth = MAX_INT
        elif isinstance(self.max_depth, int):
            max_depth = self.max_depth
        else:
            raise ValueError("The maximum depth of the tree must be either 'None' or 'int', got '{}'".format(type(self.max_depth).__name__))

        # Infer the minimum number of samples to split
        if isinstance(self.min_samples_split, int):
            min_samples_split = self.min_samples_split
        elif isinstance(self.min_samples_split, float):
            min_samples_split = int(self.min_samples_split * self.n_samples_)
        else:
            raise ValueError("The minimum number of samples to split must be either 'int' or 'float', got '{}'".format(type(self.min_samples_split).__name__))

        # Infer the value for the maximum number of features to be considered
        if self.max_features == "sqrt":
            max_features = max(1, int(np.sqrt(self.n_features_)))
        elif self.max_features == "log2":
            max_features = max(1, int(np.log2(self.n_features_)))
        elif isinstance(self.max_features, int):
            max_features = max(1, self.max_features)
        elif isinstance(self.max_features, float):
            max_features = max(1, int(self.max_features * self.n_features_))
        elif self.max_features == None:
            max_features = self.n_features_
        else:
            raise ValueError("The maximum number of features to consider must be either 'str' (log2' or 'sqrt), 'int', 'float', or 'None', got '{}'".format(type(self.max_features).__name__))

        # Obtain the Criterion object
        self.criterion_ = (CRITERIA[self.criterion](bucket       = OBOP_ALGORITHMS[self.bucket],
                                                    beta         = self.beta,
                                                    require_obop = self.criterion in CRITERIA_REQUIRING_OBOP,
                                                    random_state = random_state)
                                                    .init(self.n_classes_))

        # Obtain the Splitter object
        self.splitter_ = SPLITTERS[self.splitter](criterion    = self.criterion_,
                                                  max_splits   = self.max_splits,
                                                  random_state = random_state)

        # Obtain the Builder object
        self.builder_ = BUILDERS[self.algorithm](splitter          = self.splitter_,
                                                 max_depth         = max_depth,
                                                 min_samples_split = min_samples_split,
                                                 max_features      = max_features,
                                                 random_state      = random_state)

        # Get the indexes that sort the attributes using NumPy, since a custom Fortran version
        # is implemented, being more efficient than the C and C++ versions
        sorted_indexes = np.argsort(X.T)

        # Initialize the global precedences matrix
        gbl_precedences = np.zeros((self.n_classes_, self.n_classes_, 2), dtype = np.float64)

        # Initialize the individual precedences and pair order matrices
        ind_precedences         = np.zeros((self.n_samples_, self.n_classes_, self.n_classes_, 2), dtype = np.float64)
        ind_pair_order_matrices = np.zeros((self.n_samples_, self.n_classes_, self.n_classes_),    dtype = np.float64)

        # Moreover, initialize the consensus bucket order for the root of the tree, to avoid segmentation faults
        # when defining in Cython
        consensus = np.zeros(self.n_classes_, dtype = np.intp)

        # Normalize the sample weight for the builder
        normalize_sample_weight(sample_weight)

        # Build the tree
        self.tree_ = self.builder_.build(X                       = X,
                                         Y                       = Y,
                                         gbl_precedences         = gbl_precedences,
                                         ind_precedences         = ind_precedences,
                                         ind_pair_order_matrices = ind_pair_order_matrices,
                                         consensus               = consensus,
                                         sample_weight           = sample_weight,
                                         sorted_indexes          = sorted_indexes)

        # Return the current object already trained
        return self

    def predict(self,
                X,
                check_input = True):
        """
            Predict bucket orders for X.
        
            Parameters
            ----------
                X: numpy.ndarray
                    The test input samples.

                check_input: boolean (default = True)
                    Allow to bypass several input checking.

            Returns
            -------
                predictions: numpy.ndarray
                    Prediction for each input instance.
        """
        # Check if the model is fitted
        check_is_fitted(self, "tree_")

        # Check the input parameters (if corresponds)
        if check_input:
            X = check_X(X)
            check_n_features(X.shape[1], self.n_features_)

        # Initialize the number of samples
        n_samples = X.shape[0]

        # Initialize the predictions
        predictions = np.zeros((n_samples, self.n_classes_), dtype = np.intp)

        # Predict using the underlying tree structure
        self.tree_.predict(X, predictions)

        # Return the obtained predictions
        return predictions
