"""Base I/O code for all datasets."""


# =====================================================================
# Imports
# =====================================================================

# Standard
from os.path import join, dirname, isfile

# Third party
import numpy as np

# Local application
from ..utils.bunch import Bunch


# =====================================================================
# Methods
# =====================================================================

def load_data(module_path, data_filename):
    """Loads data from module_path/data/problem/data_type/data_filename.

    Parameters
    ----------
    module_path : str
        The module path.

    data_filename : str
        The name csv file to be loaded from
        module_path/data/problem/data_type/data_filename.csv.
        For example, "iris.csv".

    Returns
    -------
    data_lr : {None, np.ndarray} of shape (n_samples, n_features)
        A 2-D array with each row representing one Label Ranking sample
        and each column representing the features of a given sample.

    ranks_lr : {None, np.ndarray} of shape (n_samples, n_classes)
        A 2-D array with each row representing one Label Ranking target
        and each column representing the classes of a given target.

    data_plr : {None, np.ndarray} of shape (n_samples, n_features)
        A 2-D array with each row representing one Partial Label Ranking
        sample and each column representing the features of a given sample.

    ranks_plr : {None, np.ndarray} of shape (n_samples, n_classes)
        A 2-D array with each row representing one Partial Label Ranking
        target and each column representing the classes of a given target.
    """
    # Initialize the data and the rankings to None values
    (data_lr, ranks_lr, data_plr, ranks_plr) = (None, None, None, None)

    # Obtain the path to the data and the rankings
    # for Label Ranking and Partial Label Ranking
    data_filename_lr = join(
        module_path,
        "data", "label_ranking", "ranks",
        data_filename)

    data_filename_plr = join(
        module_path,
        "data", "partial_label_ranking", "ranks",
        data_filename)

    # Check whether the data and the rankings exists
    # for Label Ranking and Partial Label Ranking
    exists_ranks_lr = isfile(data_filename_lr)
    exists_ranks_plr = isfile(data_filename_plr)

    # Gather the header of the file with the
    # name of the attributes and the classes
    if exists_ranks_lr:
        header = list(np.genfromtxt(
            data_filename_lr,
            delimiter=",", max_rows=1, dtype=np.str))
    else:
        header = list(np.genfromtxt(
            data_filename_plr,
            delimiter=",", max_rows=1, dtype=np.str))

    # Extract the number of features and the
    # number of classes using a regular expression
    (n_features, n_classes) = (
        len(list(filter(lambda x: x.startswith("A"), header))),
        len(list(filter(lambda x: x.startswith("L"), header))))

    # Finally, read the .csv files
    # with the data and the rankings
    if exists_ranks_lr:
        data_lr = np.genfromtxt(
            data_filename_lr,
            delimiter=",", skip_header=True)
        (data_lr, ranks_lr) = (
            data_lr[:, :n_features],
            data_lr[:, -n_classes:].astype(np.int64))
    if exists_ranks_plr:
        data_plr = np.genfromtxt(
            data_filename_plr,
            delimiter=",", skip_header=True)
        (data_plr, ranks_plr) = (
            data_plr[:, :n_features],
            data_plr[:, -n_classes:].astype(np.int64))

    # Return the data and the rankings
    return (data_lr, ranks_lr, data_plr, ranks_plr)


def load_authorship():
    """Load and return the authorship dataset.

    The authorship dataset is a classic classification dataset adapted to
    the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                   841
    #attributes                   70
    #classes                       4
    #rankings (LR)                17
    #rankings (PLR)                5
    ===============   ==============

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_authorship
    >>> data = load_authorship()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[1, 2, 4, 3],
           [1, 2, 4, 3],
           [1, 3, 4, 2]])
    >>> data.ranks_plr[[10, 25, 50]]
    array([[1, 2, 2, 2],
           [1, 2, 2, 2],
           [1, 2, 2, 2]])
    >>> data.class_names
    ['austen', 'london', 'milton', 'shakespeare']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="authorship.csv")

    # Initialize the meaning of the features
    feature_names = [
        "a", "all", "also", "an", "and", "any", "are", "as", "at",
        "be", "been", "but", "by", "can", "do", "down", "even", "every",
        "for", "from", "had", "has", "have", "her", "his", "if", "in",
        "into", "is", "it", "its", "may", "more", "must", "my", "no",
        "not", "now", "of", "on", "one", "only", "or", "our", "should",
        "so", "some", "such", "than", "that", "the", "their", "then",
        "there", "things", "this", "to", "up", "upon", "was", "were",
        "what", "when", "which", "who", "will", "with", "would", "your",
        "book_id"
    ]

    # Initialize the meaning of the classes
    class_names = ["austen", "london", "milton", "shakespeare"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_bodyfat():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_bodyfat
    >>> data = load_bodyfat()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[7, 6, 2, 1, 5, 3, 4],
           [3, 7, 5, 2, 6, 1, 4],
           [1, 5, 3, 2, 7, 6, 4]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3',
     'class_4', 'class_5', 'class_6']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="bodyfat.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2",
        "feature_3", "feature_4", "feature_5", "feature_6"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "class_0", "class_1", "class_2", "class_3",
        "class_4", "class_5", "class_6"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_blocks():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_blocks
    >>> data = load_blocks()
    >>> data.ranks_lr
    >>> data.ranks_plr[[10, 25, 50]]
    array([[1, 2, 2, 2, 2],
           [1, 2, 2, 2, 2],
           [1, 2, 2, 2, 2]])
    >>> data.class_names
    ['text', 'horiz_line', 'graphic', 'vert_line', 'picture']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="blocks.csv")

    # Initialize the meaning of the features
    feature_names = [
        "height", "width", "area", "eccen", "p_black", "p_and",
        "mean_tr", "blackpix", "blackand", "wb_trans"
    ]

    # Initialize the meaning of the classes
    class_names = ["text", "horiz_line", "graphic", "vert_line", "picture"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_breast():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_breast
    >>> data = load_breast()
    >>> data.ranks_lr
    >>> data.ranks_plr[[10, 25, 50]]
    array([[1, 2, 2, 2, 2, 2],
           [2, 1, 2, 2, 2, 2],
           [2, 3, 1, 3, 3, 3]])
    >>> data.class_names
    ['car', 'fad', 'mas', 'gla', 'con', 'adi']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="breast.csv")

    # Initialize the meaning of the features
    feature_names = [
        "I0", "PA500", "HFS", "DA", "AREA", "A/DA", "MAX", "DR", "P"
    ]

    # Initialize the meaning of the classes
    class_names = ["car", "fad", "mas", "gla", "con", "adi"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_calhousing():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_calhousing
    >>> data = load_calhousing()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[1, 4, 2, 3],
           [3, 2, 4, 1],
           [2, 3, 1, 4]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="calhousing.csv")

    # Initialize the meaning of the features
    feature_names = ["feature_0", "feature_1", "feature_2", "feature_3"]

    # Initialize the meaning of the classes
    class_names = ["class_0", "class_1", "class_2", "class_3"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_cold():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

    References
    ----------
    .. [1] `E. Hüllermeier and J. Fürnkranz and W. Cheng and K. Brinker,
            "Label ranking by learning pairwise preferences",
            Artificial Intelligence, vol. 172, pp. 1897-1916, 2008.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_cold
    >>> data = load_cold()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[1, 3, 2, 4],
           [4, 2, 1, 3],
           [4, 2, 1, 3]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="cold.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3",
        "feature_4", "feature_5", "feature_6", "feature_7",
        "feature_8", "feature_9", "feature_10", "feature_11",
        "feature_12", "feature_13", "feature_14", "feature_15",
        "feature_16", "feature_17", "feature_18", "feature_19",
        "feature_20", "feature_21", "feature_22", "feature_23"
    ]

    # Initialize the meaning of the classes
    class_names = ["class_0", "class_1", "class_2", "class_3"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_cpu():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_cpu
    >>> data = load_cpu()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[4, 1, 2, 3, 5],
           [4, 1, 2, 3, 5],
           [3, 5, 4, 1, 2]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="cpu.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2",
        "feature_3", "feature_4", "feature_5"
    ]

    # Initialize the meaning of the classes
    class_names = ["class_0", "class_1", "class_2", "class_3", "class_4"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_diau():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

    References
    ----------
    .. [1] `E. Hüllermeier and J. Fürnkranz and W. Cheng and K. Brinker,
            "Label ranking by learning pairwise preferences",
            Artificial Intelligence, vol. 172, pp. 1897-1916, 2008.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_diau
    >>> data = load_diau()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[2, 1, 3, 5, 4, 7, 6],
           [2, 3, 1, 4, 5, 6, 7],
           [2, 3, 6, 1, 4, 5, 7]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3',
     'class_4', 'class_5', 'class_6']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="diau.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3",
        "feature_4", "feature_5", "feature_6", "feature_7",
        "feature_8", "feature_9", "feature_10", "feature_11",
        "feature_12", "feature_13", "feature_14", "feature_15",
        "feature_16", "feature_17", "feature_18", "feature_19",
        "feature_20", "feature_21", "feature_22", "feature_23"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "class_0", "class_1", "class_2", "class_3",
        "class_4", "class_5", "class_6"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_dtt():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

    References
    ----------
    .. [1] `E. Hüllermeier and J. Fürnkranz and W. Cheng and K. Brinker,
            "Label ranking by learning pairwise preferences",
            Artificial Intelligence, vol. 172, pp. 1897-1916, 2008.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_dtt
    >>> data = load_dtt()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[3, 4, 1, 2],
           [1, 2, 4, 3],
           [1, 2, 4, 3]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="dtt.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3",
        "feature_4", "feature_5", "feature_6", "feature_7",
        "feature_8", "feature_9", "feature_10", "feature_11",
        "feature_12", "feature_13", "feature_14", "feature_15",
        "feature_16", "feature_17", "feature_18", "feature_19",
        "feature_20", "feature_21", "feature_22", "feature_23"
    ]

    # Initialize the meaning of the classes
    class_names = ["class_0", "class_1", "class_2", "class_3"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_ecoli():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_ecoli
    >>> data = load_ecoli()
    >>> data.ranks_lr
    >>> data.ranks_plr[[10, 25, 50]]
    array([[1, 2, 2, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2, 2, 2]])
    >>> data.class_names
    ['cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="ecoli.csv")

    # Initialize the meaning of the features
    feature_names = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]

    # Initialize the meaning of the classes
    class_names = ["cp", "im", "pp", "imU", "om", "omL", "imL", "imS"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_elevators():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_elevators
    >>> data = load_elevators()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[9, 8, 7, 6, 1, 3, 2, 4, 5],
           [8, 4, 3, 2, 9, 6, 5, 7, 1],
           [9, 8, 7, 6, 1, 3, 2, 4, 5]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3', 'class_4',
     'class_5', 'class_6', 'class_7', 'class_8']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="elevators.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3",
        "feature_4", "feature_5", "feature_6", "feature_7", "feature_8"
    ]

    # Initialize the meaning of the classes
    class_names = ["class_0", "class_1", "class_2", "class_3",
                   "class_4", "class_5", "class_6", "class_7", "class_8"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_fried():
    """Load and return the fried dataset.

    The fried dataset is a classic regression
    dataset adapted to the Label Ranking problem.

    ===============   ==============
    #instances                 40769
    #attributes                    9
    #classes                       5
    #rankings (LR)               120
    #rankings (PLR)                -
    ===============   ==============

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_fried
    >>> data = load_fried()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[4, 2, 3, 5, 1],
           [3, 5, 1, 2, 4],
           [5, 1, 2, 4, 3]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="fried.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3",
        "feature_4", "feature_5", "feature_6", "feature_7", "feature_8"
    ]

    # Initialize the meaning of the classes
    class_names = ["class_0", "class_1", "class_2", "class_3", "class_4"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_glass():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_glass
    >>> data = load_glass()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[1, 3, 2, 4, 5, 6],
           [1, 3, 2, 4, 5, 6],
           [1, 2, 3, 4, 5, 6]])
    >>> data.ranks_plr[[10, 25, 50]]
    array([[1, 2, 3, 3, 3, 3],
           [1, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2]])
    >>> data.class_names
    ['building_windows_float_processed',
     'building_windows_non_float_processed',
     'vehicle_windows_float_processed',
     'containers', 'tableware', 'headlamps']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="glass.csv")

    # Initialize the meaning of the features
    feature_names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

    # Initialize the meaning of the classes
    class_names = [
        "building_windows_float_processed",
        "building_windows_non_float_processed",
        "vehicle_windows_float_processed",
        "containers", "tableware", "headlamps"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_heat():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

    References
    ----------
    .. [1] `E. Hüllermeier and J. Fürnkranz and W. Cheng and K. Brinker,
            "Label ranking by learning pairwise preferences",
            Artificial Intelligence, vol. 172, pp. 1897-1916, 2008.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_heat
    >>> data = load_heat()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[6, 5, 1, 4, 3, 2],
           [1, 6, 3, 5, 4, 2],
           [1, 3, 4, 5, 2, 6]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="heat.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3",
        "feature_4", "feature_5", "feature_6", "feature_7",
        "feature_8", "feature_9", "feature_10", "feature_11",
        "feature_12", "feature_13", "feature_14", "feature_15",
        "feature_16", "feature_17", "feature_18", "feature_19",
        "feature_20", "feature_21", "feature_22", "feature_23"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "class_0", "class_1", "class_2",
        "class_3", "class_4", "class_5"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_housing():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_housing
    >>> data = load_housing()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[2, 1, 4, 5, 6, 3],
           [2, 3, 6, 5, 1, 4],
           [5, 1, 3, 6, 4, 2]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="housing.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2",
        "feature_3", "feature_4", "feature_5"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "class_0", "class_1", "class_2",
        "class_3", "class_4", "class_5"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_iris():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_iris
    >>> data = load_iris()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[1, 2, 3],
           [1, 2, 3],
           [3, 1, 2]])
    >>> data.ranks_plr[[10, 25, 50]]
    array([[1, 2, 2],
           [1, 2, 2],
           [2, 1, 2]])
    >>> data.class_names
    ['setosa', 'versicolor', 'virginica']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="iris.csv")

    # Initialize the meaning of the features
    feature_names = [
        "sepal_length", "sepal_width",
        "petal_length", "petal_width"
    ]

    # Initialize the meaning of the classes
    class_names = ["setosa", "versicolor", "virginica"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_letter():
    """Load and return the letter dataset.

    The letter dataset is a classic classification dataset
    adapted to the Partial Label Ranking problem.

    ===============   ==============
    #instances                 20000
    #attributes                   16
    #classes                      26
    #rankings (LR)                 -
    #rankings (PLR)              273
    ===============   ==============

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_letter
    >>> data = load_letter()
    >>> data.ranks_lr
    >>> data.ranks_plr[[10, 25, 50]]
    array([[1, 2, 2, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2, 2, 2]])
    >>> data.class_names
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="letter.csv")

    # Initialize the meaning of the features
    feature_names = [
        "x-box", "y-box", "width", "high", "onpix",
        "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
        "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_libras():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_libras
    >>> data = load_libras()
    >>> data.ranks_lr
    >>> data.ranks_plr[[10, 25, 50]]
    array([[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           [2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           [2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    >>> data.class_names
    ['curved_swing', 'horizontal_swing', 'vertical_swing',
     'anti-clockwise_arc', 'clockwise_arc', 'circle',
     'horizontal_straight-line', 'vertical_straight-line',
     'horizontal_zigzag', 'vertical_zigzag', 'horizontal_wavy',
     'vertical_wavy', 'face-up_curve', 'face-down_curve', 'tremble']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="libras.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3", "feature_4",
        "feature_5", "feature_6", "feature_7", "feature_8", "feature_9",
        "feature_10", "feature_11", "feature_12", "feature_13", "feature_14",
        "feature_15", "feature_16", "feature_17", "feature_18", "feature_19",
        "feature_20", "feature_21", "feature_22", "feature_23", "feature_24",
        "feature_25", "feature_26", "feature_27", "feature_28", "feature_29",
        "feature_30", "feature_31", "feature_32", "feature_33", "feature_34",
        "feature_35", "feature_36", "feature_37", "feature_38", "feature_39",
        "feature_40", "feature_41", "feature_42", "feature_43", "feature_44",
        "feature_45", "feature_46", "feature_47", "feature_48", "feature_49",
        "feature_50", "feature_51", "feature_52", "feature_53", "feature_54",
        "feature_55", "feature_56", "feature_57", "feature_58", "feature_59",
        "feature_60", "feature_61", "feature_62", "feature_63", "feature_64",
        "feature_65", "feature_66", "feature_67", "feature_68", "feature_69",
        "feature_70", "feature_71", "feature_72", "feature_73", "feature_74",
        "feature_75", "feature_76", "feature_77", "feature_78", "feature_79",
        "feature_80", "feature_81", "feature_82", "feature_83", "feature_84",
        "feature_85", "feature_86", "feature_87", "feature_88", "feature_89"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "curved_swing", "horizontal_swing", "vertical_swing",
        "anti-clockwise_arc", "clockwise_arc", "circle",
        "horizontal_straight-line", "vertical_straight-line",
        "horizontal_zigzag", "vertical_zigzag", "horizontal_wavy",
        "vertical_wavy", "face-up_curve", "face-down_curve", "tremble"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_pendigits():
    """Load and return the pendigits dataset.

    The pendigits dataset is a classic classification dataset adapted to
    the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                 10992
    #attributes                   16
    #classes                      10
    #rankings (LR)              2081
    #rankings (PLR)               60
    ===============   ==============

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_pendigits
    >>> data = load_pendigits()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[ 7,  2,  8,  6,  4,  3, 10,  9,  5,  1],
           [ 5,  4,  9,  8,  1,  3, 10,  7,  6,  2],
           [ 8,  2, 10,  1,  7,  4,  9,  5,  6,  3]])
    >>> data.ranks_plr[[10, 25, 50]]
    array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
           [2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
           [2, 2, 2, 1, 2, 2, 2, 2, 2, 2]])
    >>> data.class_names
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="pendigits.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3", "feature_4",
        "feature_5", "feature_6", "feature_7", "feature_8", "feature_9",
        "feature_10", "feature_11", "feature_12", "feature_13", "feature_14",
        "feature_15"
    ]

    # Initialize the meaning of the classes
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_satimage():
    """Load and return the satimage dataset.

    The satimage dataset is a classic classification dataset
    adapted to the Partial Label Ranking problem.

    ===============   ==============
    #instances                  6435
    #attributes                   36
    #classes                       6
    #rankings (LR)                 -
    #rankings (PLR)               35
    ===============   ==============

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_satimage
    >>> data = load_satimage()
    >>> data.ranks_lr
    >>> data.ranks_plr[[10, 25, 50]]
    array([[2, 2, 2, 1, 2, 2],
           [2, 2, 1, 2, 2, 2],
           [2, 2, 1, 2, 2, 2]])
    >>> data.class_names
    ['red_soil', 'cotton_crop', 'grey_soil', 'damp_grey_soil',
     'soil_with_vegetation_stubble', 'very_damp_grey_soil']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="satimage.csv")

    # Initialize the meaning of the features and the classes
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3", "feature_4",
        "feature_5", "feature_6", "feature_7", "feature_8", "feature_9",
        "feature_10", "feature_11", "feature_12", "feature_13", "feature_14",
        "feature_15", "feature_16", "feature_17", "feature_18", "feature_19",
        "feature_20", "feature_21", "feature_22", "feature_23", "feature_24",
        "feature_25", "feature_26", "feature_27", "feature_28", "feature_29",
        "feature_30", "feature_31", "feature_32", "feature_33", "feature_34",
        "feature_35"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "red_soil", "cotton_crop", "grey_soil", "damp_grey_soil",
        "soil_with_vegetation_stubble", "very_damp_grey_soil"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_segment():
    """Load and return the segment dataset.

    The segment dataset is a classic classification dataset adapted to
    the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                  2310
    #attributes                   18
    #classes                       7
    #rankings (LR)               135
    #rankings (PLR)               20
    ===============   ==============

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_segment
    >>> data = load_segment()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[3, 7, 4, 1, 2, 5, 6],
           [2, 7, 3, 4, 1, 6, 5],
           [1, 7, 4, 2, 3, 5, 6]])
    >>> data.ranks_plr[[10, 25, 50]]
    array([[3, 3, 3, 1, 2, 3, 3],
           [2, 2, 2, 2, 1, 2, 2],
           [1, 2, 2, 2, 2, 2, 2]])
    >>> data.class_names
    ['brickface', 'sky', 'foliage', 'cement', 'window', 'path', 'grass']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="segment.csv")

    # Initialize the meaning of the features
    feature_names = [
        "region-centroid-col", "region-centroid-row", "short-line-density-5",
        "short-line-density-2", "vedge-mean", "vegde-sd", "hedge-mean",
        "hedge-sd", "intensity-mean", "rawred-mean", "rawblue-mean",
        "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean",
        "value-mean", "saturation-mean", "hue-mean"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "brickface", "sky", "foliage",
        "cement", "window", "path", "grass"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_shuttle():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_shuttle
    >>> data = load_shuttle()
    >>> data.ranks_lr
    >>> data.ranks_plr[[10, 25, 50]]
    array([[1, 2, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2, 2],
           [1, 2, 2, 2, 2, 2, 2]])
    >>> data.class_names
    ['rad_flow', 'fpv_close', 'fpv_open', 'high',
     'bypass', 'bpv_close', 'bpv_open']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="shuttle.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3",
        "feature_4", "feature_5", "feature_6", "feature_7", "feature_8"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "rad_flow", "fpv_close", "fpv_open",
        "high", "bypass", "bpv_close", "bpv_open"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_spo():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

    References
    ----------
    .. [1] `E. Hüllermeier and J. Fürnkranz and W. Cheng and K. Brinker,
            "Label ranking by learning pairwise preferences",
            Artificial Intelligence, vol. 172, pp. 1897-1916, 2008.`_

    Examples
    --------
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_spo
    >>> data = load_spo()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[10,  2,  3, 11,  7,  1,  4,  8,  9,  5,  6],
           [ 6,  9,  5,  1,  4,  8,  2,  7,  3, 11, 10],
           [10, 11,  2,  3,  1,  7,  4,  8,  5,  6,  9]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3', 'class_4',
    'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="spo.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3", "feature_4",
        "feature_5", "feature_6", "feature_7", "feature_8", "feature_9",
        "feature_10", "feature_11", "feature_12", "feature_13", "feature_14",
        "feature_15", "feature_16", "feature_17", "feature_18", "feature_19",
        "feature_20", "feature_21", "feature_22", "feature_23"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "class_0", "class_1", "class_2", "class_3", "class_4",
        "class_5", "class_6", "class_7", "class_8", "class_9", "class_10"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_stock():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_stock
    >>> data = load_stock()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[1, 3, 5, 2, 4],
           [1, 3, 5, 2, 4],
           [2, 3, 5, 1, 4]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="stock.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3", "feature_4"
    ]

    # Initialize the meaning of the classes
    class_names = ["class_0", "class_1", "class_2", "class_3", "class_4"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_vehicle():
    """Load and return the vehicle dataset.

    The vehicle dataset is a classic classification dataset adapted to
    the Label Ranking problem and the Partial Label Ranking problem.

    ===============   ==============
    #instances                   846
    #attributes                   18
    #classes                       4
    #rankings (LR)                18
    #rankings (PLR)               13
    ===============   ==============

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_vehicle
    >>> data = load_vehicle()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[2, 3, 4, 1],
           [3, 4, 2, 1],
           [3, 4, 2, 1]])
    >>> data.ranks_plr[[10, 25, 50]]
    array([[1, 2, 2, 2],
           [1, 2, 2, 2],
           [2, 1, 2, 2]])
    >>> data.class_names
    ['opel', 'saab', 'bus', 'van']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="vehicle.csv")

    # Initialize the meaning of the features
    feature_names = [
        "compactness", "circularity", "distance_circularity", "radius_ratio",
        "pr.axis_aspect_ratio", "max.length_aspect_ratio", "scatter_ratio",
        "elongatedness", "pr.axis_rectangularity", "max.length_rectangularity",
        "scaled_variance_along_major_axis", "scaled_variance_minor_axis",
        "scaled_radius_of_gyration", "skewness_about_major_axis",
        "skewness_about_minor_axis", "kurtosis_about_minor_axis",
        "kurtosis_about_major_axis", "hollows_ratio"
    ]

    # Initialize the meaning of the classes
    class_names = ["opel", "saab", "bus", "van"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_vowel():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_vowel
    >>> data = load_vowel()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[ 6,  5,  9,  3,  2,  7,  8, 10,  4, 11,  1],
           [ 5,  6,  8,  3,  1,  4,  7,  9, 10, 11,  2],
           [ 8,  9, 10, 11,  2,  4,  1,  5,  3,  7,  6]])
    >>> data.ranks_plr[[10, 25, 50]]
    array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
           [2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
           [2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2]])
    >>> data.class_names
    ['hid', 'hId', 'hEd', 'hAd', 'hYd', 'had',
     'hOd', 'hod', 'hUd', 'hud', 'hed']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="vowel.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3", "feature_4",
        "feature_5", "feature_6", "feature_7", "feature_8", "feature_9"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "hid", "hId", "hEd", "hAd", "hYd",
        "had", "hOd", "hod", "hUd", "hud", "hed"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_wine():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_wine
    >>> data = load_wine()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[1, 2, 3],
           [2, 1, 3],
           [1, 2, 3]])
    >>> data.ranks_plr[[10, 25, 50]]
    array([[1, 2, 2],
           [1, 2, 2],
           [1, 2, 2]])
    >>> data.class_names
    ['class_0', 'class_1', 'class_2']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="wine.csv")

    # Initialize the meaning of the features
    feature_names = [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
        "magnesium", "total_phenols", "flavanoids",
        "nonflavanoid_phenols", "proanthocyanins", "color_intensity",
        "hue", "OD280/OD315_of_diluted_wines", "proline"
    ]

    # Initialize the meaning of the classes
    class_names = ["class_0", "class_1", "class_2"]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_wisconsin():
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

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_wisconsin
    >>> data = load_wisconsin()
    >>> data.ranks_lr[[10, 25, 50]]
    array([[ 4, 15, 13,  7,  5,  9, 14,  8, 10, 11,  1, 12,  6,  2,  3, 16],
           [11, 14, 16, 15, 13,  4, 10,  5,  6,  9,  1,  3, 12,  7,  2,  8],
           [ 1,  3,  9, 13,  5, 16,  6, 11, 15,  8,  2,  4, 10, 14,  7, 12]])
    >>> data.ranks_plr
    >>> data.class_names
    ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5',
     'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11',
     'class_12', 'class_13', 'class_14', 'class_15']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="wisconsin.csv")

    # Initialize the meaning of the features
    feature_names = [
        "feature_0", "feature_1", "feature_2", "feature_3", "feature_4",
        "feature_5", "feature_6", "feature_7", "feature_8", "feature_9",
        "feature_10", "feature_11", "feature_12", "feature_13", "feature_14",
        "feature_15"
    ]

    # Initialize the meaning of the classes
    class_names = [
        "class_0", "class_1", "class_2", "class_3", "class_4",
        "class_5", "class_6", "class_7", "class_8", "class_9",
        "class_10", "class_11", "class_12", "class_13", "class_14",
        "class_15"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)


def load_yeast():
    """Load and return the yeast dataset.

    The yeast dataset is a classic classification dataset
    adapted to the Partial Label Ranking problem.

    ===============   ==============
    #instances                  1484
    #attributes                    8
    #classes                      10
    #rankings (LR)                 -
    #rankings (PLR)               81
    ===============   ==============

    Returns
    -------
    data : Bunch
        Dictionary-like object with the data ("data_lr", "data_plr"),
        the rankings ("ranks_lr", "ranks_plr"), the meaning of the features
        ("feature_names") and the meaning of the classes ("class_names").

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
    Let us say you are interested in the samples 10, 25 and 50,
    and want to know their class name.

    >>> from sklr.datasets import load_yeast
    >>> data = load_yeast()
    >>> data.ranks_lr
    >>> data.ranks_plr[[10, 25, 50]]
    array([[3, 1, 2, 3, 3, 3, 3, 3, 3, 3],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 2, 1, 3, 3, 3, 3, 3, 3, 3]])
    >>> data.class_names
    ['cyt', 'nuc', 'mit', 'me3', 'me2', 'me1', 'exc', 'vac', 'pox', 'erl']
    """
    # Obtain the data and the rankings
    (data_lr, ranks_lr, data_plr, ranks_plr) = load_data(
        module_path=dirname(__file__),
        data_filename="yeast.csv")

    # Initialize the meaning of the features
    feature_names = ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"]

    # Initialize the meaning of the classes
    class_names = [
        "cyt", "nuc", "mit", "me3", "me2",
        "me1", "exc", "vac", "pox", "erl"
    ]

    # Return the corresponding bunch
    return Bunch(
        data_lr=data_lr, ranks_lr=ranks_lr,
        data_plr=data_plr, ranks_plr=ranks_plr,
        feature_names=feature_names, class_names=class_names)
