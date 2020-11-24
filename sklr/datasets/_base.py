"""Base code for all datasets."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from csv import reader
from os.path import dirname, join

# Third party
import numpy as np


# =============================================================================
# Constants
# =============================================================================

# Get the path to the current module to locate the datasets
MODULE_PATH = dirname(__file__)


# =============================================================================
# Functions
# =============================================================================

def load_data(module_path, problem, data_filename):
    """Load data from module_path/data/problem/data_filename.

    Parameters
    ----------
    module_path : str
        The module path.

    problem : {"label_ranking", "partial_label_ranking"}
        The problem for which the data will be loaded.

    data_filename : str
        The name of .csv file to be loaded from
        module_path/data/problem/data_filename.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2d array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding target rankings for all samples in data.
        For example, ranks[0] is the target ranking of data[0].
    """
    with open(join(module_path, "data", problem, data_filename)) \
            as csv_file:
        data_file = reader(csv_file)
        first_line = next(data_file)
        (n_samples, n_features, n_classes) = map(int, first_line)

        data = np.zeros((n_samples, n_features), dtype=np.float64)
        ranks = np.zeros((n_samples, n_classes), dtype=np.int64)

        for (sample, line) in enumerate(data_file):
            data[sample] = line[:n_features]
            ranks[sample] = line[n_features:]

    return (data, ranks)


def load_authorship(*, problem="label_ranking"):
    """Load and return the authorship dataset (classification).

    ===============   ==============
    #instances                   841
    #attributes                   70
    #classes                       4
    #rankings (LR)                17
    #rankings (PLR)               47
    #buckets (LR)                  4
    #buckets (PLR)           3.06302
    ===============   ==============

    Read more in the :ref:`User Guide <authorship_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_authorship
    >>> (_, ranks) = load_authorship(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 4, 3],
           [1, 2, 4, 3],
           [1, 3, 4, 2]])
    >>> (_, ranks) = load_authorship(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 4, 3],
           [1, 2, 3, 2],
           [1, 2, 3, 2]])
    """
    return load_data(MODULE_PATH, problem, "authorship.csv")


def load_blocks(*, problem="partial_label_ranking"):
    """Load and return the blocks dataset (classification).

    ===============   ==============
    #instances                  5472
    #attributes                   10
    #classes                       5
    #rankings (LR)                 -
    #rankings (PLR)              116
    #buckets (LR)                  -
    #buckets (PLR)           2.33662
    ===============   ==============

    Read more in the :ref:`User Guide <blocks_dataset>`.

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_blocks
    >>> (_, ranks) = load_blocks(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 4, 4, 3, 2],
           [1, 2, 3, 3, 3],
           [1, 2, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "blocks.csv")


def load_bodyfat(*, problem="label_ranking"):
    """Load and return the bodyfat dataset (regression).

    ===============   ==============
    #instances                   252
    #attributes                    7
    #classes                       7
    #rankings (LR)               236
    #rankings (PLR)                -
    #buckets (LR)                  7
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <bodyfat_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_bodyfat
    >>> (_, ranks) = load_bodyfat(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[7, 6, 2, 1, 5, 3, 4],
           [3, 7, 5, 2, 6, 1, 4],
           [1, 5, 3, 2, 7, 6, 4]])
    """
    return load_data(MODULE_PATH, problem, "bodyfat.csv")


def load_breast(*, problem="partial_label_ranking"):
    """Load and return the breast dataset (classification).

    ===============   ==============
    #instances                   106
    #attributes                    9
    #classes                       6
    #rankings (LR)                 -
    #rankings (PLR)               62
    #buckets (LR)                  -
    #buckets (PLR)           3.92453
    ===============   ==============

    Read more in the :ref:`User Guide <breast_dataset>`.

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_breast
    >>> (_, ranks) = load_breast(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 3, 2, 3, 3, 2],
           [4, 1, 2, 3, 4, 4],
           [1, 3, 2, 3, 4, 4]])
    """
    return load_data(MODULE_PATH, problem, "breast.csv")


def load_calhousing(*, problem="label_ranking"):
    """Load and return the calhousing dataset (regression).

    ===============   ==============
    #instances                 20640
    #attributes                    4
    #classes                       4
    #rankings (LR)                24
    #rankings (PLR)                -
    #buckets (LR)                  4
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <calhousing_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_calhousing
    >>> (_, ranks) = load_calhousing(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 4, 2, 3],
           [3, 2, 4, 1],
           [2, 3, 1, 4]])
    """
    return load_data(MODULE_PATH, problem, "calhousing.csv")


def load_cold(*, problem="label_ranking"):
    """Load and return the cold dataset (real-world).

    ===============   ==============
    #instances                  2465
    #attributes                   24
    #classes                       4
    #rankings (LR)                24
    #rankings (PLR)                -
    #buckets (LR)                  4
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <real_world_datasets>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_cold
    >>> (_, ranks) = load_cold(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 3, 2, 4],
           [4, 2, 1, 3],
           [4, 2, 1, 3]])
    """
    return load_data(MODULE_PATH, problem, "cold.csv")


def load_cpu(*, problem="label_ranking"):
    """Load and return the cpu dataset (regression).

    ===============   ==============
    #instances                  8192
    #attributes                    6
    #classes                       5
    #rankings (LR)               119
    #rankings (PLR)                -
    #buckets (LR)                  5
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <cpu_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_cpu
    >>> (_, ranks) = load_cpu(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[4, 1, 2, 3, 5],
           [4, 1, 2, 3, 5],
           [3, 5, 4, 1, 2]])
    """
    return load_data(MODULE_PATH, problem, "cpu.csv")


def load_diau(*, problem="label_ranking"):
    """Load and return the diau dataset (real-world).

    ===============   ==============
    #instances                  2465
    #attributes                   24
    #classes                       7
    #rankings (LR)               967
    #rankings (PLR)                -
    #buckets (LR)                  7
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <real_world_datasets>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_diau
    >>> (_, ranks) = load_diau(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[2, 1, 3, 5, 4, 7, 6],
           [2, 3, 1, 4, 5, 6, 7],
           [2, 3, 6, 1, 4, 5, 7]])
    """
    return load_data(MODULE_PATH, problem, "diau.csv")


def load_dtt(*, problem="label_ranking"):
    """Load and return the dtt dataset (real-world).

    ===============   ==============
    #instances                  2465
    #attributes                   24
    #classes                       4
    #rankings (LR)                24
    #rankings (PLR)                -
    #buckets (LR)                  4
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <real_world_datasets>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_dtt
    >>> (_, ranks) = load_dtt(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[3, 4, 1, 2],
           [1, 2, 4, 3],
           [1, 2, 4, 3]])
    """
    return load_data(MODULE_PATH, problem, "dtt.csv")


def load_ecoli(*, problem="partial_label_ranking"):
    """Load and return the ecoli dataset (classification).

    ===============   ==============
    #instances                   336
    #attributes                    7
    #classes                       8
    #rankings (LR)                 -
    #rankings (PLR)              179
    #buckets (LR)                  -
    #buckets (PLR)           4.13988
    ===============   ==============

    Read more in the :ref:`User Guide <ecoli_dataset>`.

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_ecoli
    >>> (_, ranks) = load_ecoli(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 3, 4, 4, 3, 4, 4, 2],
           [1, 3, 3, 3, 3, 3, 3, 2],
           [1, 2, 4, 4, 4, 4, 5, 3]])
    """
    return load_data(MODULE_PATH, problem, "ecoli.csv")


def load_elevators(*, problem="label_ranking"):
    """Load and return the elevators dataset (regression).

    ===============   ==============
    #instances                 16599
    #attributes                    9
    #classes                       9
    #rankings (LR)               131
    #rankings (PLR)                -
    #buckets (LR)                  9
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <elevators_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_elevators
    >>> (_, ranks) = load_elevators(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[9, 8, 7, 6, 1, 3, 2, 4, 5],
           [8, 4, 3, 2, 9, 6, 5, 7, 1],
           [9, 8, 7, 6, 1, 3, 2, 4, 5]])
    """
    return load_data(MODULE_PATH, problem, "elevators.csv")


def load_fried(*, problem="label_ranking"):
    """Load and return the fried dataset (regression).

    ===============   ==============
    #instances                 40768
    #attributes                    9
    #classes                       5
    #rankings (LR)               120
    #rankings (PLR)                -
    #buckets (LR)                  5
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <fried_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_fried
    >>> (_, ranks) = load_fried(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[4, 2, 3, 5, 1],
           [3, 5, 1, 2, 4],
           [5, 1, 2, 4, 3]])
    """
    return load_data(MODULE_PATH, problem, "fried.csv")


def load_glass(*, problem="label_ranking"):
    """Load and return the glass dataset (classification).

    ===============   ==============
    #instances                   214
    #attributes                    9
    #classes                       6
    #rankings (LR)                30
    #rankings (PLR)              105
    #buckets (LR)                  6
    #buckets (PLR)           4.08879
    ===============   ==============

    Read more in the :ref:`User Guide <glass_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_glass
    >>> (_, ranks) = load_glass(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 3, 2, 4, 5, 6],
           [1, 3, 2, 4, 5, 6],
           [1, 2, 3, 4, 5, 6]])
    >>> (_, ranks) = load_glass(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[2, 1, 3, 3, 3, 3],
           [1, 2, 3, 4, 4, 3],
           [1, 3, 2, 5, 6, 4]])
    """
    return load_data(MODULE_PATH, problem, "glass.csv")


def load_heat(*, problem="label_ranking"):
    """Load and return the heat dataset (real-world).

    ===============   ==============
    #instances                  2465
    #attributes                   24
    #classes                       6
    #rankings (LR)               622
    #rankings (PLR)                -
    #buckets (LR)                  6
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <real_world_datasets>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_heat
    >>> (_, ranks) = load_heat(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[6, 5, 1, 4, 3, 2],
           [1, 6, 3, 5, 4, 2],
           [1, 3, 4, 5, 2, 6]])
    """
    return load_data(MODULE_PATH, problem, "heat.csv")


def load_housing(*, problem="label_ranking"):
    """Load and return the housing dataset (regression).

    ===============   ==============
    #instances                   506
    #attributes                    6
    #classes                       6
    #rankings (LR)               112
    #rankings (PLR)                -
    #buckets (LR)                  6
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <housing_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_housing
    >>> (_, ranks) = load_housing(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[2, 1, 4, 5, 6, 3],
           [2, 3, 6, 5, 1, 4],
           [5, 1, 3, 6, 4, 2]])
    """
    return load_data(MODULE_PATH, problem, "housing.csv")


def load_iris(*, problem="label_ranking"):
    """Load and return the iris dataset (classification).

    ===============   ==============
    #instances                   150
    #attributes                    4
    #classes                       3
    #rankings (LR)                 5
    #rankings (PLR)                7
    #buckets (LR)                  3
    #buckets (PLR)           2.38000
    ===============   ==============

    Read more in the :ref:`User Guide <iris_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_iris
    >>> (_, ranks) = load_iris(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 3],
           [1, 2, 3],
           [3, 1, 2]])
    >>> (_, ranks) = load_iris(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 2],
           [1, 2, 2],
           [3, 1, 2]])
    """
    return load_data(MODULE_PATH, problem, "iris.csv")


def load_letter(*, problem="partial_label_ranking"):
    """Load and return the letter dataset (classification).

    ===============   ==============
    #instances                 20000
    #attributes                   16
    #classes                      26
    #rankings (LR)                 -
    #rankings (PLR)            15014
    #buckets (LR)                  -
    #buckets (PLR)           7.03260
    ===============   ==============

    Read more in the :ref:`User Guide <letter_dataset>`.

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_letter
    >>> (_, ranks) = load_letter(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[5, 6, 5, 6, 5, 6, 5, 5, 4, 6, 6, 5, 6, 6, 6, 6, 5, 6, 2, 6, 5, 6,
            6, 1, 3, 3],
           [8, 9, 1, 9, 3, 9, 2, 7, 4, 5, 9, 6, 8, 9, 5, 9, 6, 8, 3, 8, 7, 9,
            9, 8, 9, 6],
           [2, 7, 6, 8, 7, 7, 7, 8, 7, 7, 8, 7, 5, 7, 7, 7, 4, 8, 5, 7, 5, 7,
            6, 8, 1, 3]])
    """
    return load_data(MODULE_PATH, problem, "letter.csv")


def load_libras(*, problem="partial_label_ranking"):
    """Load and return the libras dataset (classification).

    ===============   ==============
    #instances                   360
    #attributes                   90
    #classes                      15
    #rankings (LR)                 -
    #rankings (PLR)              356
    #buckets (LR)                  -
    #buckets (PLR)           6.88889
    ===============   ==============

    Read more in the :ref:`User Guide <libras_dataset>`.

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_libras
    >>> (_, ranks) = load_libras(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 6, 7, 5, 7, 4, 7, 5, 3, 6, 4, 7, 6, 3],
           [3, 1, 6, 6, 4, 5, 2, 6, 3, 4, 6, 4, 6, 5, 6],
           [7, 6, 1, 7, 6, 5, 2, 4, 3, 5, 7, 5, 7, 7, 4]])
    """
    return load_data(MODULE_PATH, problem, "libras.csv")


def load_pendigits(*, problem="label_ranking"):
    """Load and return the pendigits dataset (classification).

    ===============   ==============
    #instances                 10992
    #attributes                   16
    #classes                      10
    #rankings (LR)              2081
    #rankings (PLR)             3327
    #buckets (LR)                 10
    #buckets (PLR)           3.39747
    ===============   ==============

    Read more in the :ref:`User Guide <pendigits_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_pendigits
    >>> (_, ranks) = load_pendigits(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[ 7,  2,  8,  6,  4,  3, 10,  9,  5,  1],
           [ 5,  4,  9,  8,  1,  3, 10,  7,  6,  2],
           [ 8,  2, 10,  1,  7,  4,  9,  5,  6,  3]])
    >>> (_, ranks) = load_pendigits(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
           [2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
           [2, 2, 2, 1, 2, 2, 2, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "pendigits.csv")


def load_satimage(*, problem="partial_label_ranking"):
    """Load and return the satimage dataset (classification).

    ===============   ==============
    #instances                  6435
    #attributes                   36
    #classes                       6
    #rankings (LR)                 -
    #rankings (PLR)              504
    #buckets (LR)                  -
    #buckets (PLR)           3.35649
    ===============   ==============

    Read more in the :ref:`User Guide <satimage_dataset>`.

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_satimage
    >>> (_, ranks) = load_satimage(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[4, 5, 3, 1, 4, 2],
           [3, 3, 1, 2, 3, 3],
           [3, 5, 5, 4, 1, 2]])
    """
    return load_data(MODULE_PATH, problem, "satimage.csv")


def load_segment(*, problem="label_ranking"):
    """Load and return the segment dataset (classification).

    ===============   ==============
    #instances                  2310
    #attributes                   18
    #classes                       7
    #rankings (LR)               135
    #rankings (PLR)              271
    #buckets (LR)                  7
    #buckets (PLR)           3.03074
    ===============   ==============

    Read more in the :ref:`User Guide <segment_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_segment
    >>> (_, ranks) = load_segment(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[3, 7, 4, 1, 2, 5, 6],
           [2, 7, 3, 4, 1, 6, 5],
           [1, 7, 4, 2, 3, 5, 6]])
    >>> (_, ranks) = load_segment(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[3, 5, 5, 2, 1, 4, 5],
           [2, 2, 2, 2, 1, 2, 2],
           [1, 2, 2, 2, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "segment.csv")


def load_spo(*, problem="label_ranking"):
    """Load and return the spo dataset (real-world).

    ===============   ==============
    #instances                  2465
    #attributes                   24
    #classes                      11
    #rankings (LR)              2361
    #rankings (PLR)                -
    #buckets (LR)                 11
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <real_world_datasets>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_spo
    >>> (_, ranks) = load_spo(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[10,  2,  3, 11,  7,  1,  4,  8,  9,  5,  6],
           [ 6,  9,  5,  1,  4,  8,  2,  7,  3, 11, 10],
           [10, 11,  2,  3,  1,  7,  4,  8,  5,  6,  9]])
    """
    return load_data(MODULE_PATH, problem, "spo.csv")


def load_stock(*, problem="label_ranking"):
    """Load and return the stock dataset (regression).

    ===============   ==============
    #instances                   950
    #attributes                    5
    #classes                       5
    #rankings (LR)                51
    #rankings (PLR)                -
    #buckets (LR)                  5
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <stock_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_stock
    >>> (_, ranks) = load_stock(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 3, 5, 2, 4],
           [1, 3, 5, 2, 4],
           [2, 3, 5, 1, 4]])
    """
    return load_data(MODULE_PATH, problem, "stock.csv")


def load_vehicle(*, problem="label_ranking"):
    """Load and return the vehicle dataset (classification).

    ===============   ==============
    #instances                   846
    #attributes                   18
    #classes                       4
    #rankings (LR)                18
    #rankings (PLR)               47
    #buckets (LR)                  4
    #buckets (PLR)           3.11702
    ===============   ==============

    Read more in the :ref:`User Guide <vehicle_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_vehicle
    >>> (_, ranks) = load_vehicle(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[2, 3, 4, 1],
           [3, 4, 2, 1],
           [3, 4, 2, 1]])
    >>> (_, ranks) = load_vehicle(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 3, 4, 2],
           [1, 3, 2, 3],
           [3, 1, 4, 2]])
    """
    return load_data(MODULE_PATH, problem, "vehicle.csv")


def load_vowel(*, problem="label_ranking"):
    """Load and return the vowel dataset (classification).

    ===============   ==============
    #instances                   528
    #attributes                   10
    #classes                      11
    #rankings (LR)               294
    #rankings (PLR)              504
    #buckets (LR)                 11
    #buckets (PLR)           5.73863
    ===============   ==============

    Read more in the :ref:`User Guide <vowel_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_vowel
    >>> (_, ranks) = load_vowel(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[ 6,  5,  9,  3,  2,  7,  8, 10,  4, 11,  1],
           [ 5,  6,  8,  3,  1,  4,  7,  9, 10, 11,  2],
           [ 8,  9, 10, 11,  2,  4,  1,  5,  3,  7,  6]])
    >>> (_, ranks) = load_vowel(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[7, 6, 6, 3, 2, 6, 5, 7, 4, 5, 1],
           [8, 7, 6, 1, 3, 4, 5, 8, 5, 7, 2],
           [8, 7, 8, 7, 5, 4, 1, 3, 2, 6, 7]])
    """
    return load_data(MODULE_PATH, problem, "vowel.csv")


def load_wine(*, problem="label_ranking"):
    """Load and return the wine dataset (classification).

    ===============   ==============
    #instances                   178
    #attributes                   13
    #classes                       3
    #rankings (LR)                 5
    #rankings (PLR)               11
    #buckets (LR)                  3
    #buckets (PLR)           2.67978
    ===============   ==============

    Read more in the :ref:`User Guide <wine_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_wine
    >>> (_, ranks) = load_wine(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 3],
           [2, 1, 3],
           [1, 2, 3]])
    >>> (_, ranks) = load_wine(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 2],
           [1, 2, 3],
           [1, 2, 3]])
    """
    return load_data(MODULE_PATH, problem, "wine.csv")


def load_wisconsin(*, problem="label_ranking"):
    """Load and return the wisconsin dataset (regression).

    ===============   ==============
    #instances                   194
    #attributes                   16
    #classes                      16
    #rankings (LR)               194
    #rankings (PLR)                -
    #buckets (LR)                 16
    #buckets (PLR)                 -
    ===============   ==============

    Read more in the :ref:`User Guide <wisconsin_dataset>`.

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_wisconsin
    >>> (_, ranks) = load_wisconsin(problem="label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[ 4, 15, 13,  7,  5,  9, 14,  8, 10, 11,  1, 12,  6,  2,  3, 16],
           [11, 14, 16, 15, 13,  4, 10,  5,  6,  9,  1,  3, 12,  7,  2,  8],
           [ 1,  3,  9, 13,  5, 16,  6, 11, 15,  8,  2,  4, 10, 14,  7, 12]])
    """
    return load_data(MODULE_PATH, problem, "wisconsin.csv")


def load_yeast(*, problem="partial_label_ranking"):
    """Load and return the yeast dataset (classification).

    ===============   ==============
    #instances                  1484
    #attributes                    8
    #classes                      10
    #rankings (LR)                 -
    #rankings (PLR)             1006
    #buckets (LR)                  -
    #buckets (PLR)           5.92925
    ===============   ==============

    Read more in the :ref:`User Guide <yeast_dataset>`.

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data will be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A :term:`2d array` with each row representing one :term:`sample`
        and each column representing the :term:`features` of a given
        sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2d array holding :term:`target` :term:`rankings` for all samples
        in `data`. For example, ``ranks[0]`` is the target ranking of
        ``data[0]``.

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_yeast
    >>> (_, ranks) = load_yeast(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[3, 2, 1, 6, 6, 6, 5, 4, 6, 6],
           [1, 3, 2, 5, 5, 5, 5, 4, 5, 5],
           [3, 2, 1, 6, 5, 6, 4, 5, 6, 6]])
    """
    return load_data(MODULE_PATH, problem, "yeast.csv")
