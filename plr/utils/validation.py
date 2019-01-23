"""
    This module gathers utils for checking that
    the instances follow the established criteria.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from ..exceptions import NotFittedError

# =============================================================================
# Public objects
# =============================================================================

# Methods
__all__ = ["check_arrays",
           "check_is_fitted",
           "check_is_type",
           "check_n_features",
           "check_prob_dists",
           "check_random_state",
           "check_true_pred_sample_weight",
           "check_X",
           "check_X_Y",
           "check_X_Y_sample_weight",
           "check_Y",
           "check_Y_prob_dists",
           "check_Y_sample_weight"]

# =============================================================================
# Global methods
# =============================================================================

def check_arrays(u, v, ndim = 1):
    """
        Check if the arrays are of the corresponding type and dimension.
    """
    # Check that the first array is a NumPy array of the corresponding dimensions
    if not isinstance(u, np.ndarray):
        raise TypeError("The first array must be a NumPy array, got '{}'.".format(type(u).__name__))
    if u.ndim != ndim:
        raise ValueError("The first array must be a {}-D NumPy array, got {}-D NumPy array.".format(ndim, u.ndim))
    # Check that the second array is a NumPy array of the corresponding dimensions
    if not isinstance(v, np.ndarray):
        raise TypeError("The second array must be a NumPy array, got '{}'.".format(type(v).__name__))
    if v.ndim != ndim:
        raise ValueError("The second array must be a {}-D NumPy array, got {}-D NumPy array.".format(ndim, v.ndim))
    # Check that the length is the same
    if u.shape[0] != v.shape[0]:
        raise ValueError("The length of the arrays must be the same, got u = {} and v = {}.".format(u.shape[0], v.shape[0]))
    # All the tests passed, try to convert to double
    try:
        return (u.astype(np.float64), v.astype(np.float64))
    except:
        raise ValueError("The data type of the first array ('{}') and the second ('{}') cannot be cast to double. Check the input data types.".format(u.dtype, v.dtype))

def check_is_fitted(instance, desired_property):
    """
        Check if the instance has been fitted or not according to the desired property.
    """
    if not hasattr(instance, desired_property):
        raise NotFittedError("The '{}' instance is not fitted yet. Call 'fit' with the appropiate parameters.".format(type(instance).__name__))

def check_is_type(instance, desired_type):
    """
        Check if the instance is of the desired type or not.
    """
    if not isinstance(instance, desired_type):
        raise TypeError("The instance is not type '{}', got '{}'".format(desired_type.__name__, type(instance).__name__))

def check_n_features(current_n_features, desired_n_features):
    """
        Check that the current number of features are the desired one.
    """
    if current_n_features != desired_n_features:
        raise ValueError("Number of features of the model must match the input. Model number of features is '{}' and input number of features is '{}'".format(current_n_features, desired_n_features))

def check_prob_dists(prob_dists, ndim = 2):
    """
        Check and ensure that the probability distributions are of the corresponding type, dimension and sum one.
    """
    # Check that the probability distributions are a NumPy array of the corresponding dimension and that they sum one
    if not isinstance(prob_dists, np.ndarray):
        raise TypeError("The probability distributions must be a NumPy array, got '{}'.".format(type(prob_dists).__name__))
    if prob_dists.ndim != ndim:
        raise ValueError("The probability distributions must be a {}-D NumPy array, got {}-D NumPy array.".format(ndim, prob_dists.ndim))
    if not np.all(np.isclose(np.sum(prob_dists, axis = 1), 1)):
        raise ValueError("The probability distributions must sum one. Check the input parameters.")
    # All the tests passed, directly convert to double
    # since at this stage, all can be cast to this dtype
    return prob_dists.astype(np.float64)

def check_random_state(seed):
	"""
		Check the seed obtaining a np.random.RandomState instance.
	"""
	return seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)

def check_true_pred_sample_weight(true, pred, sample_weight, desired_dtype, true_pred_ndim = 2, sample_weight_ndim = 1):
    """
        Check that the true and predicted values for the metrics are right according to the given sample weight.
    """
    # Check that the ground truth is a NumPy array of the corresponding dimensions
    if not isinstance(true, np.ndarray):
        raise TypeError("The ground truth must be a NumPy array, got '{}'.".format(type(true).__name__))
    if true.ndim != true_pred_ndim:
        raise ValueError("The ground truth must be a {}-D NumPy array, got {}-D NumPy array.".format(true_pred_ndim, true.ndim))
    # # Check that the predictions is a NumPy array of the corresponding dimensions
    if not isinstance(pred, np.ndarray):
        raise TypeError("The predictions must be a NumPy array, got '{}'.".format(type(pred).__name__))
    if pred.ndim != true_pred_ndim:
        raise ValueError("The predictions must be a {}-D NumPy array, got {}-D NumPy array.".format(true_pred_ndim, pred.ndim))
    # Ensure that the sample weight is a NumPy array of the corresponding dimensions
    if isinstance(sample_weight, type(None)):
        sample_weight = np.ones(true.shape[0], dtype = np.float64)
    if not isinstance(sample_weight, np.ndarray):
        raise TypeError("The sample weight must be a NumPy array, got '{}'.".format(type(sample_weight).__name__))
    if sample_weight.ndim != sample_weight_ndim:
        raise ValueError("The sample weight must be a {}-D NumPy array, got {}-D NumPy array.".format(sample_weight_ndim, sample_weight.ndim))
    if np.all(sample_weight == 0):
        raise ValueError("At least one of the weights must be greater than zero. Check the input parameters.")
    # Check that the length is the same
    if true.shape[0] != pred.shape[0] or true.shape[0] != sample_weight.shape[0] or pred.shape[0] != sample_weight.shape[0]:
        raise ValueError("The length of the ground truth, predictions and sample weight must be the same, got ground_truth = {}, predictions = {} and sample_weight = {}.".format(true.shape[0], pred.shape[0], sample_weight.shape[0]))
    # All the tests passed, try to convert to double
    try:
        return (true.astype(desired_dtype), pred.astype(desired_dtype), sample_weight.astype(np.float64))
    except:
        raise ValueError("The data type of the ground truth ('{}') and the predictions ('{}') cannot be cast to double. Check the input data types.".format(true.dtype, pred.dtype))

def check_X(X, ndim = 2):
    """
        Check and ensure that the attributes are of the corresponding type and dimension.
    """
    # Check that the attributes is a NumPy array of the corresponding dimensions
    if not isinstance(X, np.ndarray):
        raise TypeError("The attributes must be a NumPy array, got '{}'.".format(type(X).__name__))
    if X.ndim != ndim:
        raise ValueError("The attributes must be a {}-D NumPy array, got {}-D NumPy array.".format(ndim, X.ndim))
    # All the tests passed, try to convert to double
    try:
        return X.astype(np.float64)
    except:
        raise ValueError("The data type of the attributes ('{}') cannot be cast to double. Check the input data types.".format(X.dtype))

def check_X_Y(X, Y, X_ndim = 2, Y_ndim = 2):
    """
        Check and ensure that the attributes are of the corresponding type and dimension.
    """
    # Check that the attributes is a NumPy array of the corresponding dimensions
    if not isinstance(X, np.ndarray):
        raise TypeError("The attributes must be a NumPy array, got '{}'.".format(type(X).__name__))
    if X.ndim != X_ndim:
        raise ValueError("The attributes must be a {}-D NumPy array, got {}-D NumPy array.".format(X_ndim, X.ndim))
    # Check that the bucket orders is a NumPy array of the corresponding dimensions
    if not isinstance(Y, np.ndarray):
        raise TypeError("The bucket orders must be a NumPy array, got '{}'.".format(type(Y).__name__))
    if Y.ndim != Y_ndim:
        raise ValueError("The bucket orders must be a {}-D NumPy array, got {}-D NumPy array.".format(Y_ndim, Y.ndim))
    # Check that the length is the same
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The length of the attributes and bucket orders must be the same, got X = {} and Y = {}.".format(X.shape[0], Y.shape[0]))
    # All the tests passed, try to convert to double
    try:
        return (X.astype(np.float64), Y.astype(np.float64))
    except:
        raise ValueError("The data type of the attributes ('{}') and bucket orders ({}) cannot be cast to double. Check the input data types.".format(X.dtype, Y.dtype))

def check_X_Y_sample_weight(X, Y, sample_weight, X_ndim = 2, Y_ndim = 2, sample_weight_ndim = 1):
    """
        Check and ensure that the attributes, bucket orders and the sample weight are of the corresponding type and dimension.
    """
    # Check that the attributes is a NumPy array of the corresponding dimensions
    if not isinstance(X, np.ndarray):
        raise TypeError("The attributes must be a NumPy array, got '{}'.".format(type(X).__name__))
    if X.ndim != X_ndim:
        raise ValueError("The attributes must be a {}-D NumPy array, got {}-D NumPy array.".format(X_ndim, X.ndim))
    # Check that the bucket orders is a NumPy array of the corresponding dimensions
    if not isinstance(Y, np.ndarray):
        raise TypeError("The bucket orders must be a NumPy array, got '{}'.".format(type(Y).__name__))
    if Y.ndim != Y_ndim:
        raise ValueError("The bucket orders must be a {}-D NumPy array, got {}-D NumPy array.".format(Y_ndim, Y.ndim))
    # Ensure that the sample weight is a NumPy array of the corresponding dimensions
    if isinstance(sample_weight, type(None)):
        sample_weight = np.ones(Y.shape[0], dtype = np.float64)
    if not isinstance(sample_weight, np.ndarray):
        raise TypeError("The sample weight must be a NumPy array, got '{}'.".format(type(sample_weight).__name__))
    if sample_weight.ndim != sample_weight_ndim:
        raise ValueError("The sample weight must be a {}-D NumPy array, got {}-D NumPy array.".format(sample_weight_ndim, sample_weight.ndim))
    if np.all(sample_weight == 0):
        raise ValueError("At least one of the weights must be greater than zero. Check the input parameters.")
    # Check that the length is the same
    if X.shape[0] != Y.shape[0] or X.shape[0] != sample_weight.shape[0] or Y.shape[0] != sample_weight.shape[0]:
        raise ValueError("The length of the attributes, bucket orders and sample weight must be the same, got X = {}, Y = {} and sample_weight = {}.".format(X.shape[0], Y.shape[0], sample_weight.shape[0]))
    # All the tests passed, try to convert to double
    try:
        return (X.astype(np.float64), Y.astype(np.float64), sample_weight.astype(np.float64))
    except:
        raise ValueError("The data type of the attributes ('{}'), bucket orders ('{}') and sample weight ('{}') cannot be cast to double. Check the input data types.".format(X.dtype, Y.dtype, sample_weight.dtype))

def check_Y(Y, ndim = 2):
    """
        Check and ensure that the bucket orders are of the corresponding type and dimension.
    """
    # Check that the bucket orders is a NumPy array of the corresponding dimensions
    if not isinstance(Y, np.ndarray):
        raise TypeError("The bucket orders must be a NumPy array, got '{}'.".format(type(Y).__name__))
    if Y.ndim != ndim:
        raise ValueError("The bucket orders must be a {}-D NumPy array, got {}-D NumPy array.".format(ndim, Y.ndim))
    # All the tests passed, try to convert to double
    try:
        return Y.astype(np.float64)
    except:
        raise ValueError("The data type of the bucket orders ('{}') cannot be cast to double. Check the input data types.".format(Y.dtype))

def check_Y_prob_dists(Y, prob_dists, Y_ndim = 2, prob_dists_ndim = 2):
    """
        Check and ensure that the bucket orders and probability distributions are of the corresponding type and dimension.
    """
    # Check that the bucket orders is a NumPy array of the corresponding dimensions
    if not isinstance(Y, np.ndarray):
        raise TypeError("The bucket orders must be a NumPy array, got '{}'.".format(type(Y).__name__))
    if Y.ndim != Y_ndim:
        raise ValueError("The bucket orders must be a {}-D NumPy array, got {}-D NumPy array.".format(Y_ndim, Y.ndim))
    # Check that the probability distributions are either None
    # or a NumPy array of the corresponding dimension and that they sum one
    if isinstance(prob_dists, type(None)):
        # All the tests passed, try to convert to double
        try:
            return (Y.astype(np.float64), prob_dists)
        except:
            raise ValueError("The data type of the bucket orders ('{}') cannot be cast to double. Check the input data types.".format(Y.dtype))
    if not isinstance(prob_dists, np.ndarray):
        raise TypeError("The probability distributions must be a NumPy array, got '{}'.".format(type(prob_dists).__name__))
    if prob_dists.ndim != prob_dists_ndim:
        raise ValueError("The probability distributions must be a {}-D NumPy array, got {}-D NumPy array.".format(prob_dists_ndim, prob_dists.ndim))
    if not np.all(np.isclose(np.sum(prob_dists, axis = 1), 1)):
        raise ValueError("The probability distributions must sum one. Check the input parameters.")
    # Check that the length is the same
    if Y.shape[0] != prob_dists.shape[0]:
        raise ValueError("The length of the bucket orders and sample weight must be the same, got Y = {} and prob_dists = {}.".format(Y.shape[0], prob_dists.shape[0]))
    # All the tests passed, try to convert to double
    try:
        return (Y.astype(np.float64), prob_dists.astype(np.float64))
    except:
        raise ValueError("The data type of the bucket orders ('{}') and sample weight ('{}') cannot be cast to double. Check the input data types.".format(Y.dtype, prob_dists.dtype))

def check_Y_sample_weight(Y, sample_weight, Y_ndim = 2, sample_weight_ndim = 1):
    """
        Check and ensure that the bucket orders and sample weight are of the corresponding type and dimension.
    """
    # Check that the bucket orders is a NumPy array of the corresponding dimensions
    if not isinstance(Y, np.ndarray):
        raise TypeError("The bucket orders must be a NumPy array, got '{}'.".format(type(Y).__name__))
    if Y.ndim != Y_ndim:
        raise ValueError("The bucket orders must be a {}-D NumPy array, got {}-D NumPy array.".format(Y_ndim, Y.ndim))
    # Ensure that the sample weight is a NumPy array of the corresponding dimensions
    if isinstance(sample_weight, type(None)):
        sample_weight = np.ones(Y.shape[0], dtype = np.float64)
    if not isinstance(sample_weight, np.ndarray):
        raise TypeError("The sample weight must be a NumPy array, got '{}'.".format(type(sample_weight).__name__))
    if sample_weight.ndim != sample_weight_ndim:
        raise ValueError("The sample weight must be a {}-D NumPy array, got {}-D NumPy array.".format(sample_weight_ndim, sample_weight.ndim))
    if np.all(sample_weight == 0):
        raise ValueError("At least one of the weights must be greater than zero. Check the input parameters.")
    # Check that the length is the same
    if Y.shape[0] != sample_weight.shape[0]:
        raise ValueError("The length of the bucket orders and sample weight must be the same, got Y = {} and sample_weight = {}.".format(Y.shape[0], sample_weight.shape[0]))
    # All the tests passed, try to convert to double
    try:
        return (Y.astype(np.float64), sample_weight.astype(np.float64))
    except:
        raise ValueError("The data type of the bucket orders ('{}') and sample weight ('{}') cannot be cast to double. Check the input data types.".format(Y.dtype, sample_weight.dtype))
