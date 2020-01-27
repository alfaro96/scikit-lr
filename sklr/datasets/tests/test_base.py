"""Testing for the base methods to load popular reference datasets."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import pytest

# Local application
from sklr.datasets import (
    load_authorship, load_bodyfat, load_blocks, load_breast, load_calhousing,
    load_cold, load_cpu, load_diau, load_dtt, load_ecoli, load_elevators,
    load_fried, load_glass, load_heat, load_housing, load_iris, load_letter,
    load_libras, load_pendigits, load_satimage, load_segment, load_shuttle,
    load_spo, load_stock, load_vehicle, load_vowel, load_wine, load_wisconsin,
    load_yeast)
from sklr.utils import unique_rankings


# =============================================================================
# Initialization
# =============================================================================

# The problems from which the data can be loaded
LABEL_RANKING = ["label_ranking"]
PARTIAL_LABEL_RANKING = ["partial_label_ranking"]

# Merge the previous lists for those
# datasets that support both problems
BOTH_PROBLEMS = [*LABEL_RANKING, *PARTIAL_LABEL_RANKING]


# =============================================================================
# Testing
# =============================================================================

def check_data(method, problem, shape):
    """Check the shape of the data provided by the method."""
    # Store the interesting attributes
    # of the dataset in a bunch object
    bunch = method(problem, return_X_Y=False)

    # Load the data and the rankings in different arrays
    (data, ranks) = method(problem, return_X_Y=True)

    # Assert that the number of samples stored
    # in the bunch object and the arrays is correct
    assert bunch.data.shape[0] == data.shape[0] == \
        ranks.shape[0] == shape[0]

    # Assert that the number of features stored
    # in the bunch object and the array is correct
    assert bunch.data.shape[1] == data.shape[1] == \
        len(bunch.feature_names) == shape[1]

    # Assert that the number of classes stored
    # in the bunch object and the array is correct
    assert bunch.ranks.shape[1] == ranks.shape[1] == \
        len(bunch.class_names) == shape[2]

    # Assert that the number of different rankings computed
    # from the bunch object and the array is correct
    assert unique_rankings(bunch.ranks).shape[0] == \
        unique_rankings(ranks).shape[0] == \
        shape[3 if problem == "label_ranking" else 4]


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_authorship(problem):
    """Test the load_authorship method."""
    check_data(load_authorship, problem, (841, 70, 4, 17, 5))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_bodyfat(problem):
    """Test the load_bodyfat method."""
    check_data(load_bodyfat, problem, (252, 7, 7, 236, None))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_blocks(problem):
    """Test the load_blocks method."""
    check_data(load_blocks, problem, (5472, 10, 5, None, 28))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_breast(problem):
    """Test the load_breast method."""
    check_data(load_breast, problem, (106, 9, 6, None, 22))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_calhousing(problem):
    """Test the load_calhousing method."""
    check_data(load_calhousing, problem, (20640, 4, 4, 24, None))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_cold(problem):
    """Test the load_cold method."""
    check_data(load_cold, problem, (2465, 24, 4, 24, None))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_cpu(problem):
    """Test the load_cpu method."""
    check_data(load_cpu, problem, (8192, 6, 5, 119, None))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_diau(problem):
    """Test the load_diau method."""
    check_data(load_diau, problem, (2465, 24, 7, 967, None))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_dtt(problem):
    """Test the load_dtt method."""
    check_data(load_dtt, problem, (2465, 24, 4, 24, None))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_ecoli(problem):
    """Test the load_ecoli method."""
    check_data(load_ecoli, problem, (336, 7, 8, None, 39))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_elevators(problem):
    """Test the load_elevators method."""
    check_data(load_elevators, problem, (16599, 9, 9, 131, None))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_fried(problem):
    """Test the load_fried method."""
    check_data(load_fried, problem, (40768, 9, 5, 120, None))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_glass(problem):
    """Test the load_glass method."""
    check_data(load_glass, problem, (214, 9, 6, 30, 23))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_heat(problem):
    """Test the load_heat method."""
    check_data(load_heat, problem, (2465, 24, 6, 622, None))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_housing(problem):
    """Test the load_housing method."""
    check_data(load_housing, problem, (506, 6, 6, 112, None))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_iris(problem):
    """Test the load_iris method."""
    check_data(load_iris, problem, (150, 4, 3, 5, 6))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_letter(problem):
    """Test the load_letter method."""
    check_data(load_letter, problem, (20000, 16, 26, None, 273))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_libras(problem):
    """Test the load_libras method."""
    check_data(load_libras, problem, (360, 90, 15, None, 38))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_pendigits(problem):
    """Test the load_pendigits method."""
    check_data(load_pendigits, problem, (10992, 16, 10, 2081, 60))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_satimage(problem):
    """Test the load_satimage method."""
    check_data(load_satimage, problem, (6435, 36, 6, None, 35))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_segment(problem):
    """Test the load_segment method."""
    check_data(load_segment, problem, (2310, 18, 7, 135, 20))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_shuttle(problem):
    """Test the load_shuttle method."""
    check_data(load_shuttle, problem, (43500, 9, 7, None, 18))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_spo(problem):
    """Test the load_spo method."""
    check_data(load_spo, problem, (2465, 24, 11, 2361, None))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_stock(problem):
    """Test the load_stock method."""
    check_data(load_stock, problem, (950, 5, 5, 51, None))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_vehicle(problem):
    """Test the load_vehicle method."""
    check_data(load_vehicle, problem, (846, 18, 4, 18, 13))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_vowel(problem):
    """Test the load_vowel method."""
    check_data(load_vowel, problem, (528, 10, 11, 294, 23))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_wine(problem):
    """Test the load_wine method."""
    check_data(load_wine, problem, (178, 13, 3, 5, 5))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_wisconsin(problem):
    """Test the load_wisconsin method."""
    check_data(load_wisconsin, problem, (194, 16, 16, 194, None))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_yeast(problem):
    """Test the load_yeast method."""
    check_data(load_yeast, problem, (1484, 8, 10, None, 81))
