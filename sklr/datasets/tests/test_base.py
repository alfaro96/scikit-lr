"""Testing of the base code for all datasets."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import pytest

# Local application
from sklr.datasets import (load_authorship, load_bodyfat, load_blocks,
                           load_breast, load_calhousing, load_cold, load_cpu,
                           load_diau, load_dtt, load_ecoli, load_elevators,
                           load_fried, load_glass, load_heat, load_housing,
                           load_iris, load_letter, load_libras, load_pendigits,
                           load_satimage, load_segment, load_spo, load_stock,
                           load_vehicle, load_vowel, load_wine, load_wisconsin,
                           load_yeast)


# =============================================================================
# Constants
# =============================================================================

LABEL_RANKING = ["label_ranking"]
PARTIAL_LABEL_RANKING = ["partial_label_ranking"]
BOTH_PROBLEMS = [*LABEL_RANKING, *PARTIAL_LABEL_RANKING]


# =============================================================================
# Functions
# =============================================================================

def check_data(load_data, problem, shape):
    """Check the shape of the data to load."""
    (data, ranks) = load_data(problem=problem)

    n_samples = shape[0]
    n_features = shape[1]
    n_classes = shape[2]

    assert data.shape[0] == ranks.shape[0] == n_samples
    assert data.shape[1] == n_features
    assert ranks.shape[1] == n_classes


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_authorship(problem):
    """Test the load_authorship function."""
    check_data(load_authorship, problem, (841, 70, 4))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_blocks(problem):
    """Test the load_blocks function."""
    check_data(load_blocks, problem, (5472, 10, 5))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_bodyfat(problem):
    """Test the load_bodyfat function."""
    check_data(load_bodyfat, problem, (252, 7, 7))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_breast(problem):
    """Test the load_breast function."""
    check_data(load_breast, problem, (106, 9, 6))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_calhousing(problem):
    """Test the load_calhousing function."""
    check_data(load_calhousing, problem, (20640, 4, 4))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_cold(problem):
    """Test the load_cold function."""
    check_data(load_cold, problem, (2465, 24, 4))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_cpu(problem):
    """Test the load_cpu function."""
    check_data(load_cpu, problem, (8192, 6, 5))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_diau(problem):
    """Test the load_diau function."""
    check_data(load_diau, problem, (2465, 24, 7))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_dtt(problem):
    """Test the load_dtt function."""
    check_data(load_dtt, problem, (2465, 24, 4))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_ecoli(problem):
    """Test the load_ecoli function."""
    check_data(load_ecoli, problem, (336, 7, 8))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_elevators(problem):
    """Test the load_elevators function."""
    check_data(load_elevators, problem, (16599, 9, 9))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_fried(problem):
    """Test the load_fried function."""
    check_data(load_fried, problem, (40768, 9, 5))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_glass(problem):
    """Test the load_glass function."""
    check_data(load_glass, problem, (214, 9, 6))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_heat(problem):
    """Test the load_heat function."""
    check_data(load_heat, problem, (2465, 24, 6))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_housing(problem):
    """Test the load_housing function."""
    check_data(load_housing, problem, (506, 6, 6))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_iris(problem):
    """Test the load_iris function."""
    check_data(load_iris, problem, (150, 4, 3))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_letter(problem):
    """Test the load_letter function."""
    check_data(load_letter, problem, (20000, 16, 26))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_libras(problem):
    """Test the load_libras function."""
    check_data(load_libras, problem, (360, 90, 15))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_pendigits(problem):
    """Test the load_pendigits function."""
    check_data(load_pendigits, problem, (10992, 16, 10))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_satimage(problem):
    """Test the load_satimage function."""
    check_data(load_satimage, problem, (6435, 36, 6))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_segment(problem):
    """Test the load_segment function."""
    check_data(load_segment, problem, (2310, 18, 7))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_spo(problem):
    """Test the load_spo function."""
    check_data(load_spo, problem, (2465, 24, 11))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_stock(problem):
    """Test the load_stock function."""
    check_data(load_stock, problem, (950, 5, 5))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_vehicle(problem):
    """Test the load_vehicle function."""
    check_data(load_vehicle, problem, (846, 18, 4))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_vowel(problem):
    """Test the load_vowel function."""
    check_data(load_vowel, problem, (528, 10, 11))


@pytest.mark.parametrize("problem", BOTH_PROBLEMS)
def test_load_wine(problem):
    """Test the load_wine function."""
    check_data(load_wine, problem, (178, 13, 3))


@pytest.mark.parametrize("problem", LABEL_RANKING)
def test_load_wisconsin(problem):
    """Test the load_wisconsin function."""
    check_data(load_wisconsin, problem, (194, 16, 16))


@pytest.mark.parametrize("problem", PARTIAL_LABEL_RANKING)
def test_load_yeast(problem):
    """Test the load_yeast function."""
    check_data(load_yeast, problem, (1484, 8, 10))
