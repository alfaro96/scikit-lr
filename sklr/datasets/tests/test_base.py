"""Testing for the base methods."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from os.path import dirname

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
# Testing
# =============================================================================

def check_shape_lr(data,
                   shape_data_lr=None,
                   shape_ranks_lr=None,
                   shape_unique_rankings_lr=None):
    """Method for checking the shape of Label Ranking datasets."""
    # Assert the number of samples and
    # the number of features for the data
    if data.data_lr is not None:
        assert data.data_lr.shape == shape_data_lr
    else:
        assert data.data_lr is None

    # Assert the number of samples and the
    # number of classes for the rankings
    if data.ranks_lr is not None:
        assert data.ranks_lr.shape == shape_ranks_lr
    else:
        assert data.ranks_lr is None

    # Assert the number of different rankings
    if data.ranks_lr is not None:
        assert (
            unique_rankings(data.ranks_lr).shape[0] ==
            shape_unique_rankings_lr)
    else:
        assert data.ranks_lr is None


def check_shape_plr(data,
                    shape_data_plr=None,
                    shape_ranks_plr=None,
                    shape_unique_rankings_plr=None):
    """Method for checking the shape of Partial Label Ranking datasets."""
    # Assert the number of samples and
    # the number of features for the data
    if data.data_plr is not None:
        assert data.data_plr.shape == shape_data_plr
    else:
        assert data.data_plr is None

    # Assert the number of samples and the
    # number of classes for the rankings
    if data.ranks_plr is not None:
        assert data.ranks_plr.shape == shape_ranks_plr
    else:
        assert data.ranks_plr is None

    # Assert the number of different rankings
    if data.ranks_plr is not None:
        assert (
            unique_rankings(data.ranks_plr).shape[0] ==
            shape_unique_rankings_plr)
    else:
        assert data.ranks_plr is None


@pytest.mark.load_authorship
def test_load_authorship():
    """Test the load_authorship method."""
    # Obtain the bunch with the the authorship dataset
    authorship = load_authorship()

    # Check the data and the rankings of the authorship dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=authorship,
        shape_data_lr=(841, 70),
        shape_ranks_lr=(841, 4),
        shape_unique_rankings_lr=17)

    # Partial Label Ranking
    check_shape_plr(
        data=authorship,
        shape_data_plr=(841, 70),
        shape_ranks_plr=(841, 4),
        shape_unique_rankings_plr=5)


@pytest.mark.load_bodyfat
def test_load_bodyfat():
    """Test the load_bodyfat method."""
    # Obtain the bunch with the bodyfat dataset
    bodyfat = load_bodyfat()

    # Check the data and the rankings of the bodyfat dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=bodyfat,
        shape_data_lr=(252, 7),
        shape_ranks_lr=(252, 7),
        shape_unique_rankings_lr=236)

    # Partial Label Ranking
    check_shape_plr(data=bodyfat)


@pytest.mark.load_blocks
def test_load_blocks():
    """Test the load_blocks method."""
    # Obtain the bunch with the blocks dataset
    blocks = load_blocks()

    # Check the data and the rankings of the blocks dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(data=blocks)

    # Partial Label Ranking
    check_shape_plr(
        data=blocks,
        shape_data_plr=(5472, 10),
        shape_ranks_plr=(5472, 5),
        shape_unique_rankings_plr=28)


@pytest.mark.load_breast
def test_load_breast():
    """Test the load_breast method."""
    # Obtain the bunch with the information
    # of the authorship dataset
    breast = load_breast()

    # Check the data and the rankings of the breast dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(data=breast)

    # Partial Label Ranking
    check_shape_plr(
        data=breast,
        shape_data_plr=(106, 9),
        shape_ranks_plr=(106, 6),
        shape_unique_rankings_plr=22)


@pytest.mark.load_calhousing
def test_load_calhousing():
    """Test the load_calhousing method."""
    # Obtain the bunch with the calhousing dataset
    calhousing = load_calhousing()

    # Check the data and the rankings of the calhousing dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=calhousing,
        shape_data_lr=(20640, 4),
        shape_ranks_lr=(20640, 4),
        shape_unique_rankings_lr=24)

    # Partial Label Ranking
    check_shape_plr(data=calhousing)


@pytest.mark.load_cold
def test_load_cold():
    """Test the load_cold method."""
    # Obtain the bunch with the cold dataset
    cold = load_cold()

    # Check the data and the rankings of the cold dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=cold,
        shape_data_lr=(2465, 24),
        shape_ranks_lr=(2465, 4),
        shape_unique_rankings_lr=24)

    # Partial Label Ranking
    check_shape_plr(data=cold)


@pytest.mark.load_cpu
def test_load_cpu():
    """Test the load_cpu method."""
    # Obtain the bunch with the cpu dataset
    cpu = load_cpu()

    # Check the data and the rankings of the cpu dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=cpu,
        shape_data_lr=(8192, 6),
        shape_ranks_lr=(8192, 5),
        shape_unique_rankings_lr=119)

    # Partial Label Ranking
    check_shape_plr(data=cpu)


@pytest.mark.load_diau
def test_load_diau():
    """Test the load_diau method."""
    # Obtain the bunch with the diau dataset
    diau = load_diau()

    # Check the data and the rankings of the diau dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=diau,
        shape_data_lr=(2465, 24),
        shape_ranks_lr=(2465, 7),
        shape_unique_rankings_lr=967)

    # Partial Label Ranking
    check_shape_plr(data=diau)


@pytest.mark.load_dtt
def test_load_dtt():
    """Test the load_dtt method."""
    # Obtain the bunch with the dtt dataset
    dtt = load_dtt()

    # Check the data and the rankings of the dtt dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=dtt,
        shape_data_lr=(2465, 24),
        shape_ranks_lr=(2465, 4),
        shape_unique_rankings_lr=24)

    # Partial Label Ranking
    check_shape_plr(data=dtt)


@pytest.mark.load_ecoli
def test_load_ecoli():
    """Test the load_ecoli method."""
    # Obtain the bunch with the ecoli dataset
    ecoli = load_ecoli()

    # Check the data and the rankings of the ecoli dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(data=ecoli)

    # Partial Label Ranking
    check_shape_plr(
        data=ecoli,
        shape_data_plr=(336, 7),
        shape_ranks_plr=(336, 8),
        shape_unique_rankings_plr=39)


@pytest.mark.load_elevators
def test_load_elevators():
    """Test the load_elevators method."""
    # Obtain the bunch with the elevators dataset
    elevators = load_elevators()

    # Check the data and the rankings of the elevators dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=elevators,
        shape_data_lr=(16599, 9),
        shape_ranks_lr=(16599, 9),
        shape_unique_rankings_lr=131)

    # Partial Label Ranking
    check_shape_plr(data=elevators)


@pytest.mark.load_fried
def test_load_fried():
    """Test the load_fried method."""
    # Obtain the bunch with the fried dataset
    fried = load_fried()

    # Check the data and the rankings of the fried dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=fried,
        shape_data_lr=(40768, 9),
        shape_ranks_lr=(40768, 5),
        shape_unique_rankings_lr=120)

    # Partial Label Ranking
    check_shape_plr(data=fried)


@pytest.mark.load_glass
def test_load_glass():
    """Test the load_glass method."""
    # Obtain the bunch with the glass dataset
    glass = load_glass()

    # Check the data and the rankings of the glass dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=glass,
        shape_data_lr=(214, 9),
        shape_ranks_lr=(214, 6),
        shape_unique_rankings_lr=30)

    # Partial Label Ranking
    check_shape_plr(
        data=glass,
        shape_data_plr=(214, 9),
        shape_ranks_plr=(214, 6),
        shape_unique_rankings_plr=23)


@pytest.mark.load_heat
def test_load_heat():
    """Test the load_heat method."""
    # Obtain the bunch with the heat dataset
    heat = load_heat()

    # Check the data and the rankings of the heat dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=heat,
        shape_data_lr=(2465, 24),
        shape_ranks_lr=(2465, 6),
        shape_unique_rankings_lr=622)

    # Partial Label Ranking
    check_shape_plr(data=heat)


@pytest.mark.load_housing
def test_load_housing():
    """Test the load_housing method."""
    # Obtain the bunch with the housing dataset
    housing = load_housing()

    # Check the data and the rankings of the housing dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=housing,
        shape_data_lr=(506, 6),
        shape_ranks_lr=(506, 6),
        shape_unique_rankings_lr=112)

    # Partial Label Ranking
    check_shape_plr(data=housing)


@pytest.mark.load_iris
def test_load_iris():
    """Test the load_iris method."""
    # Obtain the bunch with the iris dataset
    iris = load_iris()

    # Check the data and the rankings of the iris dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=iris,
        shape_data_lr=(150, 4),
        shape_ranks_lr=(150, 3),
        shape_unique_rankings_lr=5)

    # Partial Label Ranking
    check_shape_plr(
        data=iris,
        shape_data_plr=(150, 4),
        shape_ranks_plr=(150, 3),
        shape_unique_rankings_plr=6)


@pytest.mark.load_letter
def test_load_letter():
    """Test the load_letter method."""
    # Obtain the bunch with the letter dataset
    letter = load_letter()

    # Check the data and the rankings of the letter dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(data=letter)

    # Partial Label Ranking
    check_shape_plr(
        data=letter,
        shape_data_plr=(20000, 16),
        shape_ranks_plr=(20000, 26),
        shape_unique_rankings_plr=273)


@pytest.mark.load_libras
def test_load_libras():
    """Test the load_libras method."""
    # Obtain the bunch with the libras dataset
    libras = load_libras()

    # Check the data and the rankings of the libras dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(data=libras)

    # Partial Label Ranking
    check_shape_plr(
        data=libras,
        shape_data_plr=(360, 90),
        shape_ranks_plr=(360, 15),
        shape_unique_rankings_plr=38)


@pytest.mark.load_pendigits
def test_load_pendigits():
    """Test the load_pendigits method."""
    # Obtain the bunch with the pendigits dataset
    pendigits = load_pendigits()

    # Check the data and the rankings of the pendigits dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=pendigits,
        shape_data_lr=(10992, 16),
        shape_ranks_lr=(10992, 10),
        shape_unique_rankings_lr=2081)

    # Partial Label Ranking
    check_shape_plr(
        data=pendigits,
        shape_data_plr=(10992, 16),
        shape_ranks_plr=(10992, 10),
        shape_unique_rankings_plr=60)


@pytest.mark.load_satimage
def test_load_satimage():
    """Test the load_satimage method."""
    # Obtain the bunch with the satimage dataset
    satimage = load_satimage()

    # Check the data and the rankings of the satimage dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(data=satimage)

    # Partial Label Ranking
    check_shape_plr(
        data=satimage,
        shape_data_plr=(6435, 36),
        shape_ranks_plr=(6435, 6),
        shape_unique_rankings_plr=35)


@pytest.mark.load_segment
def test_load_segment():
    """Test the load_segment method."""
    # Obtain the bunch with the segment dataset
    segment = load_segment()

    # Check the data and the rankings of the segment dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=segment,
        shape_data_lr=(2310, 18),
        shape_ranks_lr=(2310, 7),
        shape_unique_rankings_lr=135)

    # Partial Label Ranking
    check_shape_plr(
        data=segment,
        shape_data_plr=(2310, 18),
        shape_ranks_plr=(2310, 7),
        shape_unique_rankings_plr=20)


@pytest.mark.load_shuttle
def test_load_shuttle():
    """Test the load_shuttle method."""
    # Obtain the bunch with the shuttle dataset
    shuttle = load_shuttle()

    # Check the data and the rankings of the shuttle dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(data=shuttle)

    # Partial Label Ranking
    check_shape_plr(
        data=shuttle,
        shape_data_plr=(43500, 9),
        shape_ranks_plr=(43500, 7),
        shape_unique_rankings_plr=18)


@pytest.mark.load_spo
def test_load_spo():
    """Test the load_spo method."""
    # Obtain the bunch with the spo dataset
    spo = load_spo()

    # Check the data and the rankings of the spo dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=spo,
        shape_data_lr=(2465, 24),
        shape_ranks_lr=(2465, 11),
        shape_unique_rankings_lr=2361)

    # Partial Label Ranking
    check_shape_plr(data=spo)


@pytest.mark.load_stock
def test_load_stock():
    """Test the load_stock method."""
    # Obtain the bunch with the stock dataset
    stock = load_stock()

    # Check the data and the rankings of the stock dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=stock,
        shape_data_lr=(950, 5),
        shape_ranks_lr=(950, 5),
        shape_unique_rankings_lr=51)

    # Partial Label Ranking
    check_shape_plr(data=stock)


@pytest.mark.load_vehicle
def test_load_vehicle():
    """Test the load_vehicle method."""
    # Obtain the bunch with the vehicle dataset
    vehicle = load_vehicle()

    # Check the data and the rankings of the vehicle dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=vehicle,
        shape_data_lr=(846, 18),
        shape_ranks_lr=(846, 4),
        shape_unique_rankings_lr=18)

    # Partial Label Ranking
    check_shape_plr(
        data=vehicle,
        shape_data_plr=(846, 18),
        shape_ranks_plr=(846, 4),
        shape_unique_rankings_plr=13)


@pytest.mark.load_vowel
def test_load_vowel():
    """Test the load_vowel method."""
    # Obtain the bunch with the vowel dataset
    vowel = load_vowel()

    # Check the data and the rankings of the vowel dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=vowel,
        shape_data_lr=(528, 10),
        shape_ranks_lr=(528, 11),
        shape_unique_rankings_lr=294)

    # Partial Label Ranking
    check_shape_plr(
        data=vowel,
        shape_data_plr=(528, 10),
        shape_ranks_plr=(528, 11),
        shape_unique_rankings_plr=23)


@pytest.mark.load_wine
def test_load_wine():
    """Test the load_wine method."""
    # Obtain the bunch with the wine dataset
    wine = load_wine()

    # Check the data and the rankings of the wine dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=wine,
        shape_data_lr=(178, 13),
        shape_ranks_lr=(178, 3),
        shape_unique_rankings_lr=5)

    # Partial Label Ranking
    check_shape_plr(
        data=wine,
        shape_data_plr=(178, 13),
        shape_ranks_plr=(178, 3),
        shape_unique_rankings_plr=5)


@pytest.mark.load_wisconsin
def test_load_wisconsin():
    """Test the load_wisconsin method."""
    # Obtain the bunch with the wisconsin dataset
    wisconsin = load_wisconsin()

    # Check the data and the rankings of the wisconsin dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(
        data=wisconsin,
        shape_data_lr=(194, 16),
        shape_ranks_lr=(194, 16),
        shape_unique_rankings_lr=194)

    # Partial Label Ranking
    check_shape_plr(data=wisconsin)


@pytest.mark.load_yeast
def test_load_yeast():
    """Test the load_yeast method."""
    # Obtain the bunch with the yeast dataset
    yeast = load_yeast()

    # Check the data and the rankings of the yeast dataset
    # for the Label Ranking and Partial Label Ranking datasets

    # Label Ranking
    check_shape_lr(data=yeast)

    # Partial Label Ranking
    check_shape_plr(
        data=yeast,
        shape_data_plr=(1484, 8),
        shape_ranks_plr=(1484, 10),
        shape_unique_rankings_plr=81)
