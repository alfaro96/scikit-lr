"""
    Testing for the "partial_label_ranking" module (plr.neighbors).
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# Scipy
from scipy.spatial import KDTree

# PLR
from plr.neighbors import KNeighborsPartialLabelRanker

# Pytest
import pytest

# =============================================================================
# Initialization
# =============================================================================

# Seed
seed = 23473

# Initialize the training dataset

# Attributes
X = np.array([[0.25, 1],
              [0,    1],
              [1,    1],
              [0.25, 0],
              [0,    0]],
              dtype = np.float64)

# Bucket orders
Y = np.array([[1, 2, 3, 4, 5],
              [4, 2, 1, 5, 3],
              [1, 2, 2, 1, 1],
              [2, 3, 3, 2, 1],
              [1, 2, 3, 2, 2]],
             dtype = np.float64)

# Sample weight
sample_weight = np.array([1/5, 1/5, 1/5, 1/5, 1/5], dtype = np.float64)

# Initialize the models and the classifiers

# Brute force
model_brute = KNeighborsPartialLabelRanker(algorithm = "brute", n_neighbors  = 2, bucket = "bpa_lia_mp2", random_state = seed)                                                          
clf_brute   = model_brute.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)

# KD-Tree
model_kd_tree = KNeighborsPartialLabelRanker(algorithm = "kd_tree", n_neighbors  = 2, bucket = "bpa_lia_mp2", random_state = seed)
clf_kd_tree   = model_kd_tree.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)                                                                                         

# Initialize the input query
x = np.array([[0.25, 0.25]], dtype = np.float64)

# Initialize the instances already sorted
sorted_Y = np.array([[[2, 3, 3, 2, 1],
                      [1, 2, 3, 2, 2],
                      [1, 2, 3, 4, 5],
                      [4, 2, 1, 5, 3],
                      [1, 2, 2, 1, 1]]],
                    dtype = np.float64)

sorted_sample_weight = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype = np.float64)

# Obtain the predictions
prediction_brute   = clf_brute.predict(X = x,   check_input = False)[0]
prediction_kd_tree = clf_kd_tree.predict(X = x, check_input = False)[0]
prediction_sorted  = clf_brute.predict(X = x, Y = sorted_Y, sample_weight = sorted_sample_weight, check_input = False)[0]

# =============================================================================
# Testing for the class KNeighborsPartialLabelRanker
# =============================================================================              
@pytest.mark.ibplr
class TestKNeighborsPartialLabelRanker(object):
    """
        Test the methods of the "KNeighborsPartialLabelRanker" class.
    """

    def test_fit_brute(self):
        """
            Test the fit method with the brute-force algorithm.
        """
        # Attributes
        np.testing.assert_array_equal(X,
                                      clf_brute.X_,
                                      err_msg = "The stored attributes for the brute-force algorithm is not correct.")

        # Bucket orders
        np.testing.assert_array_equal(Y,
                                      clf_brute.Y_,
                                      err_msg = "The stored bucket orders for the brute-force algorithm is not correct.")

        # Tree data structure
        np.testing.assert_equal(isinstance(clf_brute.tree_, type(None)),
                                True,
                                err_msg = "The stored tree data structure for the brute-force algorithm is not correct.")

    def test_fit_kd_tree(self):
        """
            Test the fit method with the KD-Tree algorithm.
        """
        # Attributes
        np.testing.assert_array_equal(X,
                                      clf_kd_tree.X_,
                                      err_msg = "The stored attributes for the KD-Tree algorithm is not correct.")

        # Bucket orders
        np.testing.assert_array_equal(Y,
                                      clf_kd_tree.Y_,
                                      err_msg = "The stored bucket orders for the KD-Tree algorithm is not correct.")

        # Tree data structure
        np.testing.assert_equal(isinstance(clf_kd_tree.tree_, KDTree),
                                True,
                                err_msg = "The stored tree data structure for the KD-Tree algorithm is not correct.")

    def test_fit_check_input(self):
        """
            Test the input checking for the "fit" method of the "KNeighborsPartialLabelRanker" class.
        """
        # The attributes are not a NumPy array
        with pytest.raises(TypeError):
            KNeighborsPartialLabelRanker().fit(X = None, Y = Y)
        
        # The attributes are not a 2-D NumPy array
        with pytest.raises(ValueError):
            KNeighborsPartialLabelRanker().fit(X = X[0], Y = Y)

        # The bucket orders are not a NumPy array
        with pytest.raises(TypeError):
            KNeighborsPartialLabelRanker().fit(X = X, Y = None)

        # The bucket orders are not a 2-D NumPy array
        with pytest.raises(ValueError):
            KNeighborsPartialLabelRanker().fit(X = X, Y = Y[0])

        # The sample weight is not a NumPy array
        with pytest.raises(TypeError):
            KNeighborsPartialLabelRanker().fit(X = X, Y = Y, sample_weight = TypeError)

        # The sample weight is not a 1-D NumPy array
        with pytest.raises(ValueError):
            KNeighborsPartialLabelRanker().fit(X = X, Y = Y, sample_weight = Y)

        # All the weights are zero
        with pytest.raises(ValueError):
            KNeighborsPartialLabelRanker().fit(X = X, Y = Y, sample_weight = sample_weight * 0)

        # The attributes, bucket orders and sample weight have not the same length
        with pytest.raises(ValueError):
            KNeighborsPartialLabelRanker().fit(X = X[:4], Y = Y, sample_weight = sample_weight)
        with pytest.raises(ValueError):
            KNeighborsPartialLabelRanker().fit(X = X, Y = Y[:2], sample_weight = sample_weight[:-1])
        with pytest.raises(ValueError):
            KNeighborsPartialLabelRanker().fit(X = X, Y = Y, sample_weight = sample_weight[:-1])

        # The attributes, bucket orders and sample weight cannot be converted to double
        with pytest.raises(ValueError):
            KNeighborsPartialLabelRanker().fit(X = np.array([["Testing"]]), Y = np.array([["Coverage"]]))

        # The algorithm is unknown
        with pytest.raises(NotImplementedError):
            KNeighborsPartialLabelRanker(algorithm = None).fit(X = X, Y = Y)

    def test_predict_brute(self):
        """
            Test the prediction for the input query using the brute-force algorithm.
        """
        np.testing.assert_array_equal(prediction_brute,
                                      np.array([1, 2, 2, 1, 1], dtype = np.intp),
                                      err_msg = "The prediction for the input query using the brute-force algorithm is not correct.")

    def test_predict_kd_tree(self):
        """
            Test the prediction for the input query using the KD-Tree algorithm.
        """
        np.testing.assert_array_equal(prediction_kd_tree,
                                      np.array([1, 2, 2, 1, 1], dtype = np.intp),
                                      err_msg = "The prediction for the input query using the KD-Tree algorithm is not correct.")

    def test_predict_sorted(self):
        """
            Test the prediction for the input query with the instances already sorted.
        """
        np.testing.assert_array_equal(prediction_sorted,
                                      np.array([1, 2, 2, 1, 1], dtype = np.intp),
                                      err_msg = "The prediction for the input query with the instances already sorted is not correct.")

    def test_predict_check_input(self):
        """
            Test the input checking for the "predict" method of the "KNeighborsPartialLabelRanker" class.
        """
        # The attributes are not a NumPy array
        with pytest.raises(TypeError):
            clf_brute.predict(X = None)

        # The attributes are not a 2-D NumPy array
        with pytest.raises(ValueError):
            clf_brute.predict(X = x[0])

        # The attributes cannot be converted to double
        with pytest.raises(ValueError):
            clf_brute.predict(X = np.array([["Testing"], ["Coverage"]]))

        # The number of features disagrees
        with pytest.raises(ValueError):
            clf_brute.predict(X = x.T)

        # The sorted bucket orders are not a NumPy array
        with pytest.raises(TypeError):
            clf_brute.predict(X = x, Y = TypeError, sample_weight = TypeError)

        # The sorted bucket orders are not a 3-D NumPy array
        with pytest.raises(ValueError):
            clf_brute.predict(X = x, Y = sorted_Y[0], sample_weight = TypeError)

        # The sorted sample weight is not a NumPy array
        with pytest.raises(TypeError):
            clf_brute.predict(X = x, Y = sorted_Y, sample_weight = TypeError)

        # The sorted sample weight is not a 2-D NumPy array
        with pytest.raises(ValueError):
            clf_brute.predict(X = x, Y = sorted_Y, sample_weight = sorted_sample_weight[0])

        # All the weights are zero
        with pytest.raises(ValueError):
            clf_brute.predict(X = x, Y = sorted_Y, sample_weight = sorted_sample_weight * 0)

        # The attributes, sorted bucket orders and sorted sample weight have not the same length
        with pytest.raises(ValueError):
            clf_brute.predict(X = x[:0], Y = sorted_Y, sample_weight = sorted_sample_weight)
        with pytest.raises(ValueError):
            clf_brute.predict(X = x, Y = np.hstack((Y, Y)), sample_weight = np.hstack((sample_weight, sample_weight)))
        with pytest.raises(ValueError):
            clf_brute.predict(X = x, Y = sorted_Y, sample_weight = np.hstack((sample_weight, sample_weight)))

        # The attributes, sorted bucket orders and sorted sample weight cannot be converted to double
        with pytest.raises(ValueError):
            KNeighborsPartialLabelRanker().fit(X = np.array([["Testing"]]), Y = np.array([[["Coverage"]]]))
