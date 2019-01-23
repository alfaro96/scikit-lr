"""
    This module gathers all the estimators to solve the
    Partial Label Ranking Problem with the state-of-the art
    decision tree greedy algorithm.
"""

# =============================================================================
# Imports
# =============================================================================

# PLR
from .base import BaseDecisionTreePartialLabelRanker

# =============================================================================
# Public objects
# =============================================================================

# Classes
__all__ = ["DecisionTreePartialLabelRanker"]

# =============================================================================
# Decision tree
# =============================================================================
class DecisionTreePartialLabelRanker(BaseDecisionTreePartialLabelRanker):
    """
        A decision tree classifier for partial rankings.
    """

    def __init__(self,
                 criterion         = "distance",
                 splitter          = "binary",
                 bucket            = "bpa_lia_mp2",
                 beta              = 0.25,
                 max_depth         = None,
                 min_samples_split = 2,
                 max_features      = None,
                 max_splits        = 2,
                 random_state      = None):
        """
            Constructor of "DecisionTreePartialLabelRanker" class.

            Parameters
            ----------
                criterion: string, optional (default = "distance")
                    The function to measure the quality of a split.
                    Supported criteria are "disagreements" for the relative frequency of disagreements between the consensus bucket
                    order and the ones in the current partition, "entropy" for entropy-based taking into account the frequency of times that each pair
                    of items precedes to the others and "distance" for the mean distance between the pair order matrix of the consensus bucket order
                    and the pair order matrices of the bucket orders in the current partition.

                splitter: string, optional (default = "binary")
                    The strategy used to choose the split at each node. Supported strategy is "binary" for the
                    best binary splitter.

                bucket: string, optional (default = "bpa_lia_mp2")
                    Algorithm that will be applied to the OBOP problem, i.e., the algorithm employed
                    to aggregate the bucket orders.

                beta: float, optional (default = 0.25)
                    The parameter to decide the precedence relation of each item w.r.t. the pivot.

                max_depth: {int, None}, optional (default = None)
                    The maximum depth of the tree. If None, then nodes are expanded until
                    all leaves are pure or until all leaves contain less than
                    min_samples_split samples.

                min_samples_split: {int, float}, optional (default = 2)
                    The minimum number of samples required to split an internal node:
                        - If int, then consider "min_samples_split" as the minimum number.
                        - If float, then "min_samples_split" is a fraction and "int(min_samples_split * n_samples)" are the minimum 
                          number of samples for each split.

                max_features: {int, float, string, None}, optional (default = None)
                    The number of features to consider when looking for the best split:
                        - If int, then consider "max_features" features at each split.
                        - If float, then "max_features" is a fraction and int(max_features * n_features)
                          features are considered at each split.
                        - If "sqrt", then "max_features = sqrt(n_features)".
                        - If "log2", then "max_features = log_2(n_features)".
                        - If None, then "max_features = n_features".

                max_splits: int, optional (default = 2)
                    The number of splits considered at each partition. Ignored if
                    splitter is equal to "binary".
                
                random_state: {None, int, RandomState instance}, optional (default = None)
                    - If int, random_state is the seed used by the random number generator.
                    - If RandomState instance, random_state is the random number generator.
                    - If None, the random number generator is the RandomState instance used
                      by np.random.

            Returns
            -------
                self: DecisionTreePartialLabelRanker
                    Current object initialized.
        """
        # Call to the constructor of the parent
        super().__init__(algorithm         = "greedy",
                         criterion         = criterion,
                         splitter          = splitter,
                         bucket            = bucket,
                         beta              = beta,
                         max_depth         = max_depth,
                         min_samples_split = min_samples_split,
                         max_features      = max_features,
                         max_splits        = max_splits,
                         random_state      = random_state)
