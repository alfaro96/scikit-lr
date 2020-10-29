"""Testing of base classes for all estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from sklearn.base import BaseEstimator

# Local application
from sklr.base import LabelRankerMixin, PartialLabelRankerMixin
from sklr.base import is_label_ranker, is_partial_label_ranker


# =============================================================================
# Classes
# =============================================================================

class Foo(LabelRankerMixin, BaseEstimator):
    """Foo estimator."""

    def __init__(self, a=None, b=None):
        """Constructor."""
        self.a = a
        self.b = b


class Bar(PartialLabelRankerMixin, BaseEstimator):
    """Bar estimator."""

    def __init__(self, c=None, d=None):
        """Constructor."""
        self.c = c
        self.d = d


# =============================================================================
# Testing
# =============================================================================

class TestBase:
    """Testing of the base classes for all estimators."""

    def setup(self):
        """Setup the attributes for testing."""
        self.foo = Foo(a="foo", b="foo")
        self.bar = Bar(c="bar", d="bar")

    def test_is_label_ranker(self):
        """Test the is_label_ranker function."""
        assert is_label_ranker(self.foo)
        assert not is_label_ranker(self.bar)

    def test_is_partial_label_ranker(self):
        """Test the is_partial_label_ranker function."""
        assert is_partial_label_ranker(self.bar)
        assert not is_partial_label_ranker(self.foo)
