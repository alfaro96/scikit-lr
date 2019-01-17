"""
    This module gathers the base class for all the estimators.
"""

# =============================================================================
# Imports
# =============================================================================

# Misc
from abc import ABC, abstractmethod

# =============================================================================
# Base partial label ranker
# =============================================================================
class BasePartialLabelRanker(ABC):
    """
        Base class for partial label rankers.
    """

    @abstractmethod
    def __init__(self,
                 bucket,
                 beta,
                 random_state):
        # Initialize the hyperparameters of the current object
        self.bucket       = bucket
        self.beta         = beta
        self.random_state = random_state

    @abstractmethod
    def fit(self,
            X,
            Y,
            sample_weight,
            check_input):
        pass

    @abstractmethod
    def predict(self,
                X,
                check_input):
        pass

    def get_params(self):
        return self.__dict__

    def set_params(self,
                   **kwargs):
        # For each one of the attributes given in **kwargs, iterate
        for key in kwargs:
            # Set the attribute of the current object if it exists
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

        # Return the current object after setting the given parameters
        return self
