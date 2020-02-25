"""Testing of base classes for all estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.miss import SimpleMisser
from sklr.base import BaseEstimator
from sklr.base import is_label_ranker, is_partial_label_ranker
from sklr.neighbors import KNeighborsLabelRanker, KNeighborsPartialLabelRanker


# =============================================================================
# Classes
# =============================================================================

class Foo(BaseEstimator):
    """Foo estimator."""

    def __init__(self, a=None, b=None):
        """Constructor."""
        self.a = a
        self.b = b


class Bar(BaseEstimator):
    """Bar estimator."""

    def __init__(self, c=None, d=None):
        """Constructor."""
        self.c = c
        self.d = d


class Baz(BaseEstimator):
    """Baz estimator."""

    def __init__(self, **hyperparams):
        """Constructor."""
        pass


# =============================================================================
# Testing
# =============================================================================

class TestBaseEstimator:
    """Testing for base class for all estimators in scikit-lr."""

    def setup(self):
        """Setup the attributes for testing."""
        # Dummy estimators for common methods of all estimators
        self.foo = Foo(a="foo", b="foo")
        self.baz = Baz(e="baz", f="baz")
        self.bar = Bar(c=self.foo, d=self.foo)

        # Real estimators for particular methods of transformers and rankers
        self.misser = SimpleMisser(strategy="top")
        self.label_ranker = KNeighborsLabelRanker(n_neighbors=3)
        self.partial_label_ranker = KNeighborsPartialLabelRanker(n_neighbors=3)

        # Toy dataset to assess that the handled errors are properly raised
        self.X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        self.Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])

        # Sample weights to test that score of the rankers
        # with and without sample weighting is different
        self.sample_weight = np.array([0.1, 0.2, 0.3, 0.4])

    def test_gets_raised(self):
        """Test that some of the errors are raised."""
        with pytest.raises(ValueError):
            self.misser.fit(self.Y).transform(self.Y[:, 1:])

        with pytest.raises(ValueError):
            self.label_ranker.fit(self.X, self.Y).predict(self.X[:, 1])

        with pytest.raises(ValueError):
            self.partial_label_ranker.fit(self.X, self.Y).predict(self.X[:, 1])

    def test_get_hyperparams(self):
        """Test the get_hyperparams method."""
        assert "c__a" in self.bar.get_hyperparams(deep=True)
        assert "c__a" not in self.bar.get_hyperparams(deep=False)

        with pytest.raises(RuntimeError):
            self.baz.get_hyperparams(deep=True)

    def test_set_hyperparams(self):
        """Test the set_hyperparams method."""
        self.bar.set_hyperparams(c="bar")
        self.bar.set_hyperparams(d__a="bar")

        assert self.bar.c == "bar"
        assert self.bar.d.a == "bar"

        with pytest.raises(ValueError):
            self.foo.set_hyperparams(c="foo")

    def test_score(self):
        """Test the score method."""
        self.label_ranker.fit(self.X, self.Y)
        self.partial_label_ranker.fit(self.X, self.Y)

        assert self.label_ranker.score(self.X, self.Y) != \
            self.label_ranker.score(self.X, self.Y, self.sample_weight)

        assert self.partial_label_ranker.score(self.X, self.Y) != \
            self.partial_label_ranker.score(self.X, self.Y, self.sample_weight)

    def test_is_label_ranker(self):
        """Test the is_label_ranker method."""
        assert not is_label_ranker(self.misser)
        assert is_label_ranker(self.label_ranker)
        assert not is_label_ranker(self.partial_label_ranker)

    def test_is_partial_label_ranker(self):
        """Test the is_partial_label_ranker method."""
        assert not is_partial_label_ranker(self.misser)
        assert is_partial_label_ranker(self.partial_label_ranker)
        assert not is_partial_label_ranker(self.label_ranker)
