"""Base I/O code for all datasets."""


# =====================================================================
# Imports
# =====================================================================

# Standard
from csv import reader
from os.path import dirname, join

# Third party
import numpy as np


# =====================================================================
# Constants
# =====================================================================

# The module path is the directory
# where this file is located in
MODULE_PATH = dirname(__file__)


# =====================================================================
# Methods
# =====================================================================

def load_data(module_path, problem, data_file_name):
    """Loads data from ``module_path/data/problem/data_file_name``.

    Parameters
    ----------
    module_path : str
        The module path.

    problem : {"label_ranking", "partial_label_ranking"}
        The problem for which the data is to be loaded.

    data_file_name : str
        The name of ``.csv`` file to be loaded from
        ``module_path/data/problem/data_file_name``.
        For example '``wine.csv``'.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.
    """
    with open(join(module_path, "data", problem, data_file_name)) \
            as csv_file:
        # Read the first line of the data file to extract the
        # number of samples, the number of features and the
        # number of classes. These values are needed to know
        # the shape of the data and the shape of the rankings
        data_file = reader(csv_file)
        first_line = next(data_file)
        (n_samples, n_features, n_classes) = map(int, first_line)

        data = np.zeros((n_samples, n_features), dtype=np.float64)
        ranks = np.zeros((n_samples, n_classes), dtype=np.int64)

        for (sample, line) in enumerate(data_file):
            data[sample] = line[:n_features]
            ranks[sample] = line[n_features:]

    return (data, ranks)


def load_authorship(problem="label_ranking"):
    """Load and return the authorship dataset.

    The authorship dataset is a classic classification dataset adapted
    to the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                   841
    #attributes                   70
    #classes                       4
    #rankings (LR)                17
    #rankings (PLR)                5
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `J. S. Simonoff, Analyzing Categorical Data. Springer, 2003.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

    .. [3] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

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
    array([[1, 2, 2, 2],
           [1, 2, 2, 2],
           [1, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "authorship.csv")


def load_blocks(problem="partial_label_ranking"):
    """Load and return the blocks dataset.

    The blocks dataset is a classic classification
    dataset adapted to the Partial Label Ranking problem.

    ===============   ==============
    #instances                  5472
    #attributes                   10
    #classes                       5
    #rankings (LR)                 -
    #rankings (PLR)               28
    ===============   ==============

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `F. Esposito and D. Malerba and G. Semeraro,
            "Multistrategy Learning for Document Recognition",
            Applied Artificial Intelligence, vol. 8, pp. 33-84, 1994.`_

    .. [2] `D. Malerba and F. Esposito and G. Semeraro, "A Further Comparison
            of Simplification Methods for Decision-Tree Induction",
            Lecture Notes in Statistics, vol. 112, pp. 365-374, 1996.`_

    .. [3] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_blocks
    >>> (_, ranks) = load_blocks(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 2, 2, 2],
           [1, 2, 2, 2, 2],
           [1, 2, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "blocks.csv")


def load_bodyfat(problem="label_ranking"):
    """Load and return the bodyfat dataset.

    The bodyfat dataset is a classic regression
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                   252
    #attributes                    7
    #classes                       7
    #rankings (LR)               236
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `C. Bailey, "Smart Exercise: Burning Fat, Getting Fit".
            Houghton Mifflin, 1996.`_

    .. [2] `A. R. Behnke and J. H. Wilmore, "Evaluation and Regulation
            of Body Build and Composition". Prentice-Hall, 1974.`_

    .. [3] `W. E. Siri, "Gross composition of the body", Advances in
            Biological and Medical Physics, vol. 4, pp. 239-280, 1956.`_

    .. [4] `F. Katch and W. McArdle, "Nutrition, Weight Control, and Exercise".
            Houghton Mifflin, 1977.`_

    .. [5] `J. Wilmore, "Athletic Training and Physical Fitness:
            Physiological Principles of the Conditioning Process".
            Allyn and Bacon, 1976.`_

    .. [6] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

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


def load_breast(problem="partial_label_ranking"):
    """Load and return the breast dataset.

    The breast dataset is a classic classification
    dataset adapted to the Partial Label Ranking problem.

    ===============   ==============
    #instances                   106
    #attributes                    9
    #classes                       6
    #rankings (LR)                 -
    #rankings (PLR)               22
    ===============   ==============

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `J. Jossinet, "Variability of impedivity in normal
            and pathological breast tissue", Medical and Biological
            Engineering and Computing, vol. 34, pp. 346-350, 1996.`_

    .. [2] `J. Estrela and J. P. Marques, "Classification of breast
            tissue by electrical impedance spectroscopy", Medical and
            Biological Engineering and Computing, vol. 38, pp. 26-30, 2000.`_

    .. [3] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_breast
    >>> (_, ranks) = load_breast(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 2, 2, 2, 2],
           [2, 1, 2, 2, 2, 2],
           [2, 3, 1, 3, 3, 3]])
    """
    return load_data(MODULE_PATH, problem, "breast.csv")


def load_calhousing(problem="label_ranking"):
    """Load and return the calhousing dataset.

    The calhousing dataset is a classic regression
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                 20640
    #attributes                    4
    #classes                       4
    #rankings (LR)                24
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `R. K. Pace and R. Barry, "Sparse spatial autoregressions",
            Spatial and Probability Letters, vol. 33, pp. 291-297, 1997.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

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


def load_cold(problem="label_ranking"):
    """Load and return the cold dataset.

    The cold dataset is real-world biological
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                  2465
    #attributes                   24
    #classes                       4
    #rankings (LR)                24
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `E. Hüllermeier and J. Fürnkranz and W. Cheng and K. Brinker,
            "Label ranking by learning pairwise preferences",
            Artificial Intelligence, vol. 172, pp. 1897-1916, 2008.`_

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


def load_cpu(problem="label_ranking"):
    """Load and return the cpu dataset.

    The cpu dataset is a classic regression
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                  8192
    #attributes                    6
    #classes                       5
    #rankings (LR)               119
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `C. E. Rasmussen and R. M. Neal and G. Hinton and D. van Camp
            and M. Revow and R. Kustra and R. Tibshirani. "Data for
            Evaluating Learning in Valid Experiments",
            https://www.cs.toronto.edu/~delve/group.html, 1996.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

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


def load_diau(problem="label_ranking"):
    """Load and return the diau dataset.

    The diau dataset is a real-world biological
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                  2465
    #attributes                   24
    #classes                       7
    #rankings (LR)               967
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `E. Hüllermeier and J. Fürnkranz and W. Cheng and K. Brinker,
            "Label ranking by learning pairwise preferences",
            Artificial Intelligence, vol. 172, pp. 1897-1916, 2008.`_

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


def load_dtt(problem="label_ranking"):
    """Load and return the dtt dataset.

    The dtt dataset is a real-world biological
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    # instances                 2465
    # attributes                  24
    # classes                      4
    # rankings (LR)               24
    # rankings (PLR)               -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `E. Hüllermeier and J. Fürnkranz and W. Cheng and K. Brinker,
            "Label ranking by learning pairwise preferences",
            Artificial Intelligence, vol. 172, pp. 1897-1916, 2008.`_

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


def load_ecoli(problem="partial_label_ranking"):
    """Load and return the ecoli dataset.

    The ecoli dataset is a classic classification
    dataset adapted to the Partial Label Ranking problem.

    ===============   ==============
    #instances                   336
    #attributes                    7
    #classes                       8
    #rankings (LR)                 -
    #rankings (PLR)               39
    ===============   ==============

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `P. Horton and K. Nakai, "A Probabilistic Classification System
            for Predicting the Cellular Localization Sites of Proteins",
            In Proceedings of the Fourth International Conference on
            Intelligent Systems for Molecular Biology, 1996, pp. 109-115.`_

    .. [2] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_ecoli
    >>> (_, ranks) = load_ecoli(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 2, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "ecoli.csv")


def load_elevators(problem="label_ranking"):
    """Load and return the elevators dataset.

    The elevators dataset is a classic regression
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                 16599
    #attributes                    9
    #classes                       9
    #rankings (LR)               131
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `C. E. Rasmussen and R. M. Neal and G. Hinton and D. van Camp
            and M. Revow and R. Kustra and R. Tibshirani. "Data for
            Evaluating Learning in Valid Experiments",
            https://www.cs.toronto.edu/~delve/group.html, 1996.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

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


def load_fried(problem="label_ranking"):
    """Load and return the fried dataset.

    The fried dataset is a classic regression
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                 40768
    #attributes                    9
    #classes                       5
    #rankings (LR)               120
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `J. H. Friedman, "Multivariate Adaptive Regression Splines",
            The Annals of Statistics, vol. 19, pp. 1-67, 1991.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

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


def load_glass(problem="label_ranking"):
    """Load and return the glass dataset.

    The glass dataset is a classic classification dataset adapted to
    the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                   214
    #attributes                    9
    #classes                       6
    #rankings (LR)                30
    #rankings (PLR)               23
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `I. W. Evett and E. J. Spiehler, "Rule induction in forensic
            science", In Knowledge Based Systems, 1989, pp. 152-160.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

    .. [3] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

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
    array([[1, 2, 3, 3, 3, 3],
           [1, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "glass.csv")


def load_heat(problem="label_ranking"):
    """Load and return the heat dataset.

    The heat dataset is a real-world biological
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                  2465
    #attributes                   24
    #classes                       6
    #rankings (LR)               622
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `E. Hüllermeier and J. Fürnkranz and W. Cheng and K. Brinker,
            "Label ranking by learning pairwise preferences",
            Artificial Intelligence, vol. 172, pp. 1897-1916, 2008.`_

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


def load_housing(problem="label_ranking"):
    """Load and return the housing dataset.

    The housing dataset is a classic regression
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                   506
    #attributes                    6
    #classes                       6
    #rankings (LR)               112
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `D. Harrison and D. L. Rubinfeld, "Hedonic prices and the
            demand for clean air", Journal of Environmental Economics
            and Management, vol. 5, pp. 81-102, 1978.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

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


def load_iris(problem="label_ranking"):
    """Load and return the iris dataset.

    The iris dataset is a classic classification dataset adapted to
    the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                   150
    #attributes                    4
    #classes                       3
    #rankings (LR)                 5
    #rankings (PLR)                6
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `R. A. Fisher, "The use of multiple measurements in taxonomic
            problems", Annals of Eugenics, vol. 7, pp. 179-188, 1936.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

    .. [3] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

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
           [2, 1, 2]])
    """
    return load_data(MODULE_PATH, problem, "iris.csv")


def load_letter(problem="partial_label_ranking"):
    """Load and return the letter dataset.

    The letter dataset is a classic classification
    dataset adapted to the Partial Label Ranking problem.

    ===============   ==============
    #instances                 20000
    #attributes                   16
    #classes                      26
    #rankings (LR)                 -
    #rankings (PLR)              273
    ===============   ==============

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `P. W. Frey and D. J. Slate, "Letter recognition using
            Holland-style adaptive classifiers", Machine Learning,
            vol. 6, pp. 161-182, 1991.`_

    .. [2] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_letter
    >>> (_, ranks) = load_letter(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
        2, 2, 2, 2],
           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2],
           [2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "letter.csv")


def load_libras(problem="partial_label_ranking"):
    """Load and return the libras dataset.

    The libras dataset is a classic classification
    dataset adapted to the Partial Label Ranking problem.

    ===============   ==============
    #instances                   360
    #attributes                   90
    #classes                      15
    #rankings (LR)                 -
    #rankings (PLR)               38
    ===============   ==============

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `D. B. Dias and R. C. B. Madeo and T. Rocha and H. H. Bíscaro
            and S. M. Peres, "Hand Movement Recognition for Brazilian
            Sign Language: A Study Using Distance-based Neural Networks",
            In Proceedings of the 2009 international joint conference
            on Neural Networks, 2009, pp. 2355-2362.`_

    .. [2] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_libras
    >>> (_, ranks) = load_libras(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           [2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           [2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "libras.csv")


def load_pendigits(problem="label_ranking"):
    """Load and return the pendigits dataset.

    The pendigits dataset is a classic classification dataset adapted
    to the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                 10992
    #attributes                   16
    #classes                      10
    #rankings (LR)              2081
    #rankings (PLR)               60
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `F. Alimoglu and E. Alpaydin, "Methods of Combining Multiple
            Classifiers Based on Different Representations for Pen-based
            Handwritten Digit Recognition", In Proceedings of the Fifth
            Turkish Artificial Intelligence and Artificial Neural Networks
            Symposium, 1996.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

    .. [3] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

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


def load_satimage(problem="partial_label_ranking"):
    """Load and return the satimage dataset.

    The satimage dataset is a classic classification
    dataset adapted to the Partial Label Ranking problem.

    ===============   ==============
    #instances                  6435
    #attributes                   36
    #classes                       6
    #rankings (LR)                 -
    #rankings (PLR)               35
    ===============   ==============

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `C. Fend and A. Sutherland and S. King and S. Muggleton and R.
            Henery, "Comparison of Machine Learning Classifiers to Statistics
            and Neural Networks", In Proceedings of the 4th International
            Workshop on Artificial Intelligence and Statistics, 1993,
            pp. 41-52.`_

    .. [2] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_satimage
    >>> (_, ranks) = load_satimage(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[2, 2, 2, 1, 2, 2],
           [2, 2, 1, 2, 2, 2],
           [2, 2, 1, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "satimage.csv")


def load_segment(problem="label_ranking"):
    """Load and return the segment dataset.

    The segment dataset is a classic classification dataset adapted
    to the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                  2310
    #attributes                   18
    #classes                       7
    #rankings (LR)               135
    #rankings (PLR)               20
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `C. Fend and A. Sutherland and S. King and S. Muggleton and R.
            Henery, "Comparison of Machine Learning Classifiers to Statistics
            and Neural Networks", In Proceedings of the 4th International
            Workshop on Artificial Intelligence and Statistics, 1993,
            pp. 41-52.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

    .. [3] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

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
    array([[3, 3, 3, 1, 2, 3, 3],
           [2, 2, 2, 2, 1, 2, 2],
           [1, 2, 2, 2, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "segment.csv")


def load_shuttle(problem="partial_label_ranking"):
    """Load and return the shuttle dataset.

    The shuttle dataset is a classic classification
    dataset adapted to the Partial Label Ranking problem.

    ===============   ==============
    #instances                 43500
    #attributes                    9
    #classes                       7
    #rankings (LR)                 -
    #rankings (PLR)               18
    ===============   ==============

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `C. Fend and A. Sutherland and S. King and S. Muggleton and R.
            Henery, "Comparison of Machine Learning Classifiers to Statistics
            and Neural Networks", In Proceedings of the 4th International
            Workshop on Artificial Intelligence and Statistics, 1993,
            pp. 41-52.`_

    .. [2] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_shuttle
    >>> (_, ranks) = load_shuttle(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[1, 2, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "shuttle.csv")


def load_spo(problem="label_ranking"):
    """Load and return the spo dataset.

    The spo dataset is real-world biological
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                  2465
    #attributes                   24
    #classes                      11
    #rankings (LR)              2361
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `E. Hüllermeier and J. Fürnkranz and W. Cheng and K. Brinker,
            "Label ranking by learning pairwise preferences",
            Artificial Intelligence, vol. 172, pp. 1897-1916, 2008.`_

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


def load_stock(problem="label_ranking"):
    """Load and return the stock dataset.

    The stock dataset is a classic regression
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                   950
    #attributes                    5
    #classes                       5
    #rankings (LR)                51
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `H. Altay and I. Uysal, "Bilkent University Function Approximation
            Repository", 2000.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

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


def load_vehicle(problem="label_ranking"):
    """Load and return the vehicle dataset.

    The vehicle dataset is a classic classification dataset adapted
    to the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                   846
    #attributes                   18
    #classes                       4
    #rankings (LR)                18
    #rankings (PLR)               13
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `C. Fend and A. Sutherland and S. King and S. Muggleton and R.
            Henery, "Comparison of Machine Learning Classifiers to Statistics
            and Neural Networks", In Proceedings of the 4th International
            Workshop on Artificial Intelligence and Statistics, 1993,
            pp. 41-52.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

    .. [3] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

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
    array([[1, 2, 2, 2],
           [1, 2, 2, 2],
           [2, 1, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "vehicle.csv")


def load_vowel(problem="label_ranking"):
    """Load and return the vowel dataset.

    The vowel dataset is a classic classification dataset adapted to
    the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                   528
    #attributes                   10
    #classes                      11
    #rankings (LR)               294
    #rankings (PLR)               23
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `M. Niranjan and F. Fallside, "Neural networks and radial basis
            functions in classifying static speech patterns", Computer Speech
            and Language, vol. 4, pp. 275-289, 1990.`_

    .. [2] `S. Renals and R. Rohwer, "Phoneme Classification Experiments
            Using Radial Basis Functions", In Proceedings of the International
            1989 Joint Conference on Neural Networks, 1989, pp. 461-467.`_

    .. [3] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

    .. [4] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

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
    array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
           [2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "vowel.csv")


def load_wine(problem="label_ranking"):
    """Load and return the wine dataset.

    The wine dataset is a classic classification dataset adapted to
    the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                   178
    #attributes                   13
    #classes                       3
    #rankings (LR)                 5
    #rankings (PLR)                5
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking", "partial_label_ranking"}, \
            default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `S. Aeberhard and D. Coomans and O. de Vel, "Comparative analysis
            of statistical pattern recognition methods in high dimensional
            settings", Pattern Recognition, vol. 27, pp. 1065-1077, 1994.`_

    .. [2] `S. Aeberhard and D. Coomans and O. de Vel, "Improvements to the
            classification performance of RDA", Journal of Chemometrics,
            vol. 7, pp. 99-115, 1993.`_

    .. [3] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

    .. [4] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

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
           [1, 2, 2],
           [1, 2, 2]])
    """
    return load_data(MODULE_PATH, problem, "wine.csv")


def load_wisconsin(problem="label_ranking"):
    """Load and return the wisconsin dataset.

    The wisconsin dataset is a classic regression
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                   194
    #attributes                   16
    #classes                      16
    #rankings (LR)               194
    #rankings (PLR)                -
    ===============   ==============

    Parameters
    ----------
    problem : {"label_ranking"}, default="label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `W. N. Street and O. L. Mangasarian and W. H. Wolberg, "An inductive
            learning approach to prognostic prediction", In Proceedings of the
            Twelfth International Conference on Machine Learning, 1995,
            pp. 522-530.`_

    .. [2] `O. L. Mangasarian and W. N. Street and W. H. Wolberg,
            "Breast cancer diagnosis and prognosis via linear programming",
            Operations Research, vol. 43, pp. 570-577, 1995.`_

    .. [3] `W. H. Wolberg and W. N. Street and D. M. Heisey and
            O. L. Mangasarian, "Computerized breast cancer diagnosis and
            prognosis from fine needle aspirates",
            Archives of Surgery, vol 130, pp. 511-516, 1995.`_

    .. [4] `W. H. Wolberg and W. N. Street and O. L. Mangasarian,
            "Image analysis and machine learning applied to breast
            cancer diagnosis and prognosis", Analytical and Quantitative
            Cytology and Histology, vol. 17, pp. 77-87, 1995.`_

    .. [5] `W. H. Wolberg and W. N. Street and O. L. Mangasarian,
            "Computer-derived nuclear grade and breast cancer prognosis",
            Analytical and Quantitative Cytology and Histology,
            vol. 17, pp. 257-264, 1995.`_

    .. [6] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings of
            the 26th International Conference on Machine Learning, 2009,
            pp. 161-168.`_

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


def load_yeast(problem="partial_label_ranking"):
    """Load and return the yeast dataset.

    The yeast dataset is a classic classification
    dataset adapted to the Partial Label Ranking problem.

    ===============   ==============
    #instances                  1484
    #attributes                    8
    #classes                      10
    #rankings (LR)                 -
    #rankings (PLR)               81
    ===============   ==============

    Parameters
    ----------
    problem : {"partial_label_ranking"}, default="partial_label_ranking"
        The problem for which the data is to be loaded.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features), dtype=np.float64
        A 2-D array with each row representing one sample and each
        column representing the features of a given sample.

    ranks : ndarray of shape (n_samples, n_classes), dtype=np.int64
        A 2-D array holding target rankings for all the samples in
        ``data``. For example, ``ranks[0]`` is the target ranking
        for ``data[0]``.

    References
    ----------
    .. [1] `P. Horton and K. Nakai, "A Probablistic Classification System
            for Predicting the Cellular Localization Sites of Proteins",
            Intelligent Systems in Molecular Biology, vol. 4, pp. 109-115,
            1996.`_

    .. [2] `K. Nakai and M. Kanehisa, "Expert Sytem for Predicting Protein
            Localization Sites in Gram-Negative Bacteria", Proteins: Structure,
            Function, and Bioinformatics, vol. 11, pp. 95-110, 1991.`_

    .. [3] `K. Nakai and M. Kanehisa, "A Knowledge Base for Predicting Protein
            Localization Sites in Eukaryotic Cells", Genomics, vol. 14,
            pp. 987-911, 1992.`_

    .. [4] `J. C. Alfaro, J. A. Aledo, y J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50.

    >>> from sklr.datasets import load_yeast
    >>> (_, ranks) = load_yeast(problem="partial_label_ranking")
    >>> ranks[[10, 25, 50]]
    array([[3, 1, 2, 3, 3, 3, 3, 3, 3, 3],
           [1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           [3, 2, 1, 3, 3, 3, 3, 3, 3, 3]])
    """
    return load_data(MODULE_PATH, problem, "yeast.csv")
