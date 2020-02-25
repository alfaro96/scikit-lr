"""Testing of base classes for all estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import pytest

# Local application
from sklr.datasets import load_iris
from sklr.base import is_label_ranker, is_partial_label_ranker
from sklr.base import (BaseEstimator, LabelRankerMixin,
                       PartialLabelRankerMixin, TransformerMixin)


# =============================================================================
# Classes
# =============================================================================

class Foo(BaseEstimator, LabelRankerMixin):
    """Foo estimator."""

    def __init__(self, a=None, b=None):
        """Constructor."""
        self.a = a
        self.b = b

    def fit(self, X, Y):
        """Fit."""
        self._validate_train_data(X, Y)

        self.foo_ = "foo"

        return self


class Bar(BaseEstimator, PartialLabelRankerMixin):
    """Bar estimator."""

    def __init__(self, c=None, d=None):
        """Constructor."""
        self.c = c
        self.d = d

    def fit(self, X, Y):
        """Fit."""
        self._validate_train_data(X, Y)

        self.bar_ = "bar"

        return self


class Baz(BaseEstimator, TransformerMixin):
    """Baz estimator."""

    def __init__(self, **hyperparams):
        """Constructor."""

    def fit(self, Y):
        """Fit."""
        self._validate_train_rankings(Y)

        self.baz_ = "baz"

        return self


# =============================================================================
# Testing
# =============================================================================

class TestBaseEstimator:
    """Testing for base class for all estimators in scikit-lr."""

    def setup(self):
        """Setup the attributes for testing."""
        self.foo = Foo(a="foo", b="foo")
        self.baz = Baz(e="baz", f="baz")
        self.bar = Bar(c=self.foo, d=self.foo)

        (self.X, self.Y) = load_iris(problem="label_ranking")

    def test_gets_raised(self):
        """Test that some of the handled errors are raised."""
        with pytest.raises(ValueError, match="Number of features"):
            self.foo.fit(self.X, self.Y)._validate_test_data(self.X[:, 1:])

        with pytest.raises(ValueError, match="Number of features"):
            self.bar.fit(self.X, self.Y)._validate_test_data(self.X[:, 1:])

        with pytest.raises(ValueError, match="Number of classes"):
            self.baz.fit(self.Y)._validate_test_rankings(self.Y[:, 1:])

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

    def test_is_label_ranker(self):
        """Test the is_label_ranker method."""
        assert is_label_ranker(self.foo)
        assert not is_label_ranker(self.bar)
        assert not is_label_ranker(self.baz)

    def test_is_partial_label_ranker(self):
        """Test the is_partial_label_ranker method."""
        assert is_partial_label_ranker(self.bar)
        assert not is_partial_label_ranker(self.foo)
        assert not is_partial_label_ranker(self.baz)
