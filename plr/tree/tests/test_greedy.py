"""
    Testing for the "greedy" module (plr.tree).
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from plr.tree import DecisionTreePartialLabelRanker

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

# Entropy

# Binary
model_fully_binary_entropy = DecisionTreePartialLabelRanker(criterion = "entropy", splitter = "binary", random_state = seed)
clf_fully_binary_entropy   = model_fully_binary_entropy.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)

# Disagreements

# Binary
model_fully_binary_disagreements = DecisionTreePartialLabelRanker(criterion = "disagreements", splitter = "binary", random_state = seed)
clf_fully_binary_disagreements   = model_fully_binary_disagreements.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)

# Distance

# Binary
model_fully_binary_distance = DecisionTreePartialLabelRanker(criterion = "distance", splitter = "binary", random_state = seed)
clf_fully_binary_distance   = model_fully_binary_distance.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)

# Others

# Stump decision tree
model_stump_1 = DecisionTreePartialLabelRanker(max_depth = 1)
model_stump_2 = DecisionTreePartialLabelRanker(min_samples_split = X.shape[0])
model_stump_3 = DecisionTreePartialLabelRanker(min_samples_split = 1.0)
clf_stump_1   = model_stump_1.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)
clf_stump_2   = model_stump_2.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)
clf_stump_3   = model_stump_3.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)

# Features
model_features_1 = DecisionTreePartialLabelRanker(max_features = "sqrt")
model_features_2 = DecisionTreePartialLabelRanker(max_features = "log2")
model_features_3 = DecisionTreePartialLabelRanker(max_features = 1)
model_features_4 = DecisionTreePartialLabelRanker(max_features = 0.5)
clf_features_1   = model_features_1.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)
clf_features_2   = model_features_2.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)
clf_features_3   = model_features_3.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)
clf_features_4   = model_features_4.fit(X = X, Y = Y, sample_weight = sample_weight, check_input = False)

# Initialize the input query
x = np.array([[0.25, 0.25]], dtype = np.float64)

# Obtain the predictions

# Entropy
prediction_fully_binary_entropy = clf_fully_binary_entropy.predict(X = x, check_input = False)[0]

# Disagreements
prediction_fully_binary_disagreements = clf_fully_binary_disagreements.predict(X = x, check_input = False)[0]

# Distance
prediction_fully_binary_distance = clf_fully_binary_distance.predict(X = x, check_input = False)[0]

# Others

# Stump decision tree
prediction_stump_1 = clf_stump_1.predict(X = x, check_input = False)[0]
prediction_stump_2 = clf_stump_2.predict(X = x, check_input = False)[0]
prediction_stump_3 = clf_stump_3.predict(X = x, check_input = False)[0]

# Features
prediction_features_1 = clf_features_1.predict(X = x, check_input = False)[0]
prediction_features_2 = clf_features_2.predict(X = x, check_input = False)[0]
prediction_features_3 = clf_features_3.predict(X = x, check_input = False)[0]
prediction_features_4 = clf_features_4.predict(X = x, check_input = False)[0]

# =============================================================================
# Testing for the class DecisionTreePartialLabelRanker
# ============================================================================= 
@pytest.mark.plrt
class TestDecisionTreePartialLabelRanker(object):
    """
        Test the methods of the "DecisionTreePartialLabelRanker" class.
    """

    def test_fit_binary_entropy(self):
        """
            Test the "fit" method with the binary splitter and entropy criterion.
        """
        # Depth
        np.testing.assert_equal(clf_fully_binary_entropy.tree_.depth_,
                                3,
                                err_msg = "The depth of the tree obtained with the greedy builder, binary splitter and entropy criterion is not correct.")

        # Inner nodes
        np.testing.assert_equal(clf_fully_binary_entropy.tree_.n_inner_nodes_,
                                4,
                                err_msg = "The number of inner nodes of the tree obtained with the greedy builder, binary splitter and entropy criterion is not correct.")

        # Leaf nodes
        np.testing.assert_equal(clf_fully_binary_entropy.tree_.n_leaf_nodes_,
                                5,
                                err_msg = "The number of leaf nodes of the tree obtained with the greedy builder, binary splitter and entropy criterion is not correct.")

    def test_fit_binary_disagreements(self):
        """
            Test the "fit" method with the binary splitter and disagreements criterion.
        """
        # Depth
        np.testing.assert_equal(clf_fully_binary_disagreements.tree_.depth_,
                                3,
                                err_msg = "The depth of the tree obtained obtained with the greedy builder, binary splitter and disagreements criterion is not correct.")

        # Inner nodes
        np.testing.assert_equal(clf_fully_binary_disagreements.tree_.n_inner_nodes_,
                                4,
                                err_msg = "The number of inner nodes of the tree obtained with the greedy builder, binary splitter and disagreements criterion is not correct.")

        # Leaf nodes
        np.testing.assert_equal(clf_fully_binary_disagreements.tree_.n_leaf_nodes_,
                                5,
                                err_msg = "The number of leaf nodes of the tree obtained with the greedy builder, binary splitter and disagreements criterion is not correct.")

    def test_fit_binary_distance(self):
        """
            Test the "fit" method with the binary splitter and distance criterion.
        """
        # Depth
        np.testing.assert_equal(clf_fully_binary_distance.tree_.depth_,
                                3,
                                err_msg = "The depth of the tree obtained with the greedy builder, binary splitter and distance criterion is not correct.")

        # Inner nodes
        np.testing.assert_equal(clf_fully_binary_distance.tree_.n_inner_nodes_,
                                4,
                                err_msg = "The number of inner nodes of the tree obtained with the greedy builder, binary splitter and distance criterion is not correct.")

        # Leaf nodes
        np.testing.assert_equal(clf_fully_binary_distance.tree_.n_leaf_nodes_,
                                5,
                                err_msg = "The number of leaf nodes of the tree obtained with the greedy builder, binary splitter and distance criterion is not correct.")

    def test_fit_stump(self):
        """
            Test the "fit" method for testing the "depth" and "min_samples_split" hyperparameters.
        """
        # Stump decision trees

        # Depth
        np.testing.assert_equal(clf_stump_1.tree_.depth_,
                                1,
                                err_msg = "The depth of the stump tree is not correct.")

        # Inner nodes
        np.testing.assert_equal(clf_stump_1.tree_.n_inner_nodes_,
                                1,
                                err_msg = "The number of inner nodes of the stump tree is not correct.")

        # Leaf nodes
        np.testing.assert_equal(clf_stump_1.tree_.n_leaf_nodes_,
                                2,
                                err_msg = "The number of leaf nodes of the stump tree is not correct.")

        # Depth
        np.testing.assert_equal(clf_stump_2.tree_.depth_,
                                1,
                                err_msg = "The depth of the stump tree is not correct.")

        # Inner nodes
        np.testing.assert_equal(clf_stump_2.tree_.n_inner_nodes_,
                                1,
                                err_msg = "The number of inner nodes of the stump tree is not correct.")

        # Leaf nodes
        np.testing.assert_equal(clf_stump_2.tree_.n_leaf_nodes_,
                                2,
                                err_msg = "The number of leaf nodes of the stump tree is not correct.")

        # Depth
        np.testing.assert_equal(clf_stump_3.tree_.depth_,
                                1,
                                err_msg = "The depth of the stump tree is not correct.")

        # Inner nodes
        np.testing.assert_equal(clf_stump_3.tree_.n_inner_nodes_,
                                1,
                                err_msg = "The number of inner nodes of the stump tree is not correct.")

        # Leaf nodes
        np.testing.assert_equal(clf_stump_3.tree_.n_leaf_nodes_,
                                2,
                                err_msg = "The number of leaf nodes of the stump tree is not correct.")

    def test_fit_features(self):
        """
            Test the "fit" method for testing the "max_features" hyperparameter.
        """
        # One-feature decision trees

        # Depth
        np.testing.assert_equal(clf_features_1.tree_.depth_,
                                3,
                                err_msg = "The depth of the tree built from one feature is not correct.")

        # Inner nodes
        np.testing.assert_equal(clf_features_1.tree_.n_inner_nodes_,
                                4,
                                err_msg = "The number of inner nodes of the tree built from one feature is not correct.")

        # Leaf nodes
        np.testing.assert_equal(clf_features_1.tree_.n_leaf_nodes_,
                                5,
                                err_msg = "The number of leaf nodes of the tree built from one feature is not correct.")

        # Depth
        np.testing.assert_equal(clf_features_2.tree_.depth_,
                                3,
                                err_msg = "The depth of the tree built from one feature is not correct.")

        # Inner nodes
        np.testing.assert_equal(clf_features_2.tree_.n_inner_nodes_,
                                4,
                                err_msg = "The number of inner nodes of the tree built from one feature is not correct.")

        # Leaf nodes
        np.testing.assert_equal(clf_features_2.tree_.n_leaf_nodes_,
                                5,
                                err_msg = "The number of leaf nodes of the tree built from one feature is not correct.")

        # Depth
        np.testing.assert_equal(clf_features_3.tree_.depth_,
                                3,
                                err_msg = "The depth of the tree built from one feature is not correct.")

        # Inner nodes
        np.testing.assert_equal(clf_features_3.tree_.n_inner_nodes_,
                                4,
                                err_msg = "The number of inner nodes of the tree built from one feature is not correct.")

        # Leaf nodes
        np.testing.assert_equal(clf_features_3.tree_.n_leaf_nodes_,
                                5,
                                err_msg = "The number of leaf nodes of the tree built from one feature is not correct.")

        # Depth
        np.testing.assert_equal(clf_features_4.tree_.depth_,
                                3,
                                err_msg = "The depth of the tree built from one feature is not correct.")

        # Inner nodes
        np.testing.assert_equal(clf_features_4.tree_.n_inner_nodes_,
                                4,
                                err_msg = "The number of inner nodes of the tree built from one feature is not correct.")

        # Leaf nodes
        np.testing.assert_equal(clf_features_4.tree_.n_leaf_nodes_,
                                5,
                                err_msg = "The number of leaf nodes of the tree built from one feature is not correct.")

    def test_fit_check_input(self):
        """
            Test the input checking for the "fit" method of the "DecisionTreePartialLabelRanker" class.
        """
        # The attributes are not a NumPy array
        with pytest.raises(TypeError):
            DecisionTreePartialLabelRanker().fit(X = None, Y = Y)
        
        # The attributes are not a 2-D NumPy array
        with pytest.raises(ValueError):
            DecisionTreePartialLabelRanker().fit(X = X[0], Y = Y)

        # The bucket orders are not a NumPy array
        with pytest.raises(TypeError):
            DecisionTreePartialLabelRanker().fit(X = X, Y = None)

        # The bucket orders are not a 2-D NumPy array
        with pytest.raises(ValueError):
            DecisionTreePartialLabelRanker().fit(X = X, Y = Y[0])

        # The sample weight is not a NumPy array
        with pytest.raises(TypeError):
            DecisionTreePartialLabelRanker().fit(X = X, Y = Y, sample_weight = TypeError)

        # The sample weight is not a 1-D NumPy array
        with pytest.raises(ValueError):
            DecisionTreePartialLabelRanker().fit(X = X, Y = Y, sample_weight = Y)

        # All the weights are zero
        with pytest.raises(ValueError):
            DecisionTreePartialLabelRanker().fit(X = X, Y = Y, sample_weight = sample_weight * 0)

        # The attributes, bucket orders and sample weight have not the same length
        with pytest.raises(ValueError):
            DecisionTreePartialLabelRanker().fit(X = X[:4], Y = Y, sample_weight = sample_weight)
        with pytest.raises(ValueError):
            DecisionTreePartialLabelRanker().fit(X = X, Y = Y[:2], sample_weight = sample_weight[:-1])
        with pytest.raises(ValueError):
            DecisionTreePartialLabelRanker().fit(X = X, Y = Y, sample_weight = sample_weight[:-1])

        # The attributes, bucket orders and sample weight cannot be converted to double
        with pytest.raises(ValueError):
            DecisionTreePartialLabelRanker().fit(X = np.array([["Testing"]]), Y = np.array([["Coverage"]]))

        # The maximum depth is not "None" nor "int"
        with pytest.raises(ValueError):
            DecisionTreePartialLabelRanker(max_depth = 0.0).fit(X = X, Y = Y, sample_weight = sample_weight)

        # The minimum number of samples to split is not "int" nor "float"
        with pytest.raises(ValueError):
            DecisionTreePartialLabelRanker(min_samples_split = None).fit(X = X, Y = Y, sample_weight = sample_weight)

        # The maximum number of features is not "sqrt", "log2", "int", "float" nor "None"
        with pytest.raises(ValueError):
            DecisionTreePartialLabelRanker(max_features = "log10").fit(X = X, Y = Y, sample_weight = sample_weight)

    def test_predict_binary_entropy(self):
        """
            Test the prediction for the input query using the binary splitter and entropy criterion.
        """
        np.testing.assert_array_equal(prediction_fully_binary_entropy,
                                      np.array([2, 3, 3, 2, 1]),
                                      err_msg = "The prediction of the tree obtained with the greedy builder, binary splitter and entropy criterion is not correct.")

    def test_predict_binary_disagremeents(self):
        """
            Test the prediction for the input query using the binary splitter and disagreements criterion.
        """
        np.testing.assert_array_equal(prediction_fully_binary_disagreements,
                                      np.array([2, 3, 3, 2, 1]),
                                      err_msg = "The prediction of the tree obtained with the greedy builder, binary splitter and disagreements criterion is not correct.")

    def test_predict_distance(self):
        """
            Test the prediction for the input query using the binary splitter and distance criterion.
        """
        np.testing.assert_array_equal(prediction_fully_binary_distance,
                                      np.array([2, 3, 3, 2, 1]),
                                      err_msg = "The prediction of the tree obtained with the greedy builder, binary splitter and distance criterion is not correct.")

    def test_predict_stump(self):
        """
            Test the "predict" method for testing the "depth" and "min_samples_split" hyperparameters.
        """
        np.testing.assert_equal(prediction_stump_1,
                                np.array([1, 2, 2, 1, 1]),
                                err_msg = "The prediction of the stump tree is not correct.")

        np.testing.assert_equal(prediction_stump_2,
                                np.array([1, 2, 2, 1, 1]),
                                err_msg = "The prediction of the stump tree is not correct.")

        np.testing.assert_equal(prediction_stump_3,
                                np.array([1, 2, 2, 1, 1]),
                                err_msg = "The prediction of the stump tree is not correct.")

    def test_predict_features(self):
        """
            Test the "predict" method for testing the "max_features" hyperparameter.
        """
        np.testing.assert_equal(prediction_features_1,
                                np.array([2, 3, 3, 2, 1]),
                                err_msg = "The prediction of the tree built with one feature is not correct.")

        np.testing.assert_equal(prediction_features_2,
                                np.array([2, 3, 3, 2, 1]),
                                err_msg = "The prediction of the tree built with one feature is not correct.")

        np.testing.assert_equal(prediction_features_3,
                                np.array([2, 3, 3, 2, 1]),
                                err_msg = "The prediction of the tree built with one feature is not correct.")

        np.testing.assert_equal(prediction_features_4,
                                np.array([2, 3, 3, 2, 1]),
                                err_msg = "The prediction of the tree built with one feature is not correct.")

    def test_predict_check_input(self):
        """
            Test the input checking for the "predict" method of the "DecisionTreePartialLabelRanker" class.
        """
        # The attributes are not a NumPy array
        with pytest.raises(TypeError):
            clf_fully_binary_entropy.predict(X = None)

        # The attributes are not a 2-D NumPy array
        with pytest.raises(ValueError):
            clf_fully_binary_entropy.predict(X = x[0])

        # The attributes cannot be converted to double
        with pytest.raises(ValueError):
            clf_fully_binary_entropy.predict(X = np.array([["Testing"], ["Coverage"]]))

        # The number of features disagrees
        with pytest.raises(ValueError):
            clf_fully_binary_entropy.predict(X = x.T)
