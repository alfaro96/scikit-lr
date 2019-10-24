"""Testing for Bunch util."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import pytest

# Local application
from plr.utils import Bunch


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.bunch
def test_bunch():
    """Test the Bunch class."""
    # Initialize the Bunch object
    bunch = Bunch(a=1)

    # Assert that the value is properly returned
    assert bunch.a == 1

    # Assert that an error is raised when the attribute is not available
    with pytest.raises(AttributeError):
        bunch.b

    # Add the "b" attribute
    bunch.b = 2

    # Assert that the list of keys is properly returned
    assert dir(bunch) == ["a", "b"]
