"""Testing for Bunch util."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import pytest

# Local application
from sklr.utils import Bunch


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.bunch
def test_bunch():
    """Test the Bunch class."""
    # Initialize the Bunch object with an
    # attribute called "a" set to a value of 1
    bunch = Bunch(a=1)

    # Assert that this value is
    # properly returned by the bunch
    assert bunch.a == 1

    # Assert that an error is raised when trying to
    # access to an attribute that is not available
    with pytest.raises(AttributeError):
        bunch.b

    # Add the "b" attribute to the bunch
    # from outside of the constructor
    bunch.b = 2

    # Assert that the available keys
    # in the bunch are properly returned
    assert dir(bunch) == ["a", "b"]
