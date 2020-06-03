.. _datasets:

.. currentmodule:: sklr.datasets

=========================
Dataset loading utilities
=========================

The :mod:`sklr.datasets` module provides some toy and real-world datasets
commonly used by the Machine Learning community to benchmark algorithms.

General dataset API
===================

The main dataset interface can be used to load
:ref:`standard datasets <toy_datasets>` and
:ref:`real-world datasets <real_world_datasets>`.

These functions return a tuple ``(X, Y)`` consisting of a
``(n_samples, n_features)`` :class:`~numpy.ndarray` ``X`` and
an array of shape ``(n_samples, n_classes)`` containing the
:term:`target` :term:`rankings` ``Y``.

.. _toy_datasets:

Toy datasets
============

We provide standard datasets that can be loaded using the following functions:

.. autosummary::

    load_authorship
    load_blocks
    load_bodyfat
    load_breast
    load_calhousing
    load_cpu
    load_ecoli
    load_elevators
    load_fried
    load_glass
    load_housing
    load_iris
    load_letter
    load_libras
    load_pendigits
    load_satimage
    load_segment
    load_stock
    load_vehicle
    load_vowel
    load_wine
    load_wisconsin
    load_yeast

These datasets were obtained by transforming :term:`multiclass`
and :term:`continuous` (regression) problems of the `UCI Machine
Learning Repository`_ to the label ranking problem and the partial
label ranking problem.

.. topic:: References:

    * W. Cheng, J. Hühn and E. Hüllermeier, "Decision tree and instance-based
      learning for label ranking", In Proceedings of the 26th International
      Conference on Machine Learning, pp. 161-168, 2009.

    * J. C. Alfaro, J. A. Aledo and J. A. Gámez, "Learning decision trees for
      the Partial Label Ranking problem", International Journal of Intelligent
      Systems, 2020, Submitted.

.. _authorship_dataset:

Anacaldata authorship dataset
-----------------------------

This dataset belongs to a collection of datasets used to analyze categorical
data.

.. topic:: References:

    * J. S. Simonoff, "Analyzing Categorical Data", Springer-Verlag, 2003.

.. _blocks_dataset:

Page blocks dataset
-------------------

This dataset was obtained by a segmentation process of all the blocks of the
page layout of a document.

.. topic:: References:

    * D. Malerba, F. Esposito and G. Semeraro, "A Further Comparison of
      Simplification Methods for Decision-Tree Induction", In Learning
      from Data: Artificial Intelligence and Statistics, pp. 365-374,
      1995.

    * F. Esposito, D. Malerba and G. Semeraro, "Multistrategy Learning
      for Document Recognition", Applied Artificial Intelligence,
      vol. 8, pp. 33-84, 1994.

.. _bodyfat_dataset:

Body fat dataset
----------------

This dataset contains estimates of the percentage of body fat determined
by underwater weighting and body circumference measurements for men.

.. topic:: References:

    * A. R. Behnke and J. H. Wilmore, "Evaluation and Regulation
      of Body Build and Composition", Prentice-Hall, 1974.

    * W. E. Siri, "The Gross Composition of the Body", Advances in
      Biological and Medical Physics, vol. 4, pp. 239-280, 1956.

.. _breast_dataset:

Breast tissue dataset
---------------------

This dataset measures freshly excised breast tissues, which, plotted
in the plane, constitute the impedance spectrum from where the breast
tissue features are computed.

.. topic:: References:

    * J. Jossinet, "Variability of impedivity in normal and pathological
      breast tissue", Medical & Biological Engineering & Computing, vol. 34,
      pp. 346-350, 1996.

    * J. E. Silva, J. P. de Sá, J. Jossinet, "Classification of Breast
      Tissue by Electrical Impedance Spectroscopy", Medical & Biological
      Engineering & Computing, vol. 38, pp. 26-30, 2000.

.. _calhousing_dataset:

California housing dataset
--------------------------

This dataset was derived from the 1990 U.S. census, using one row per
census block group (a block group is the smallest geographical unit
for which the U.S. Census Bureau publishes sample data).

.. topic:: References:

    * R. K. Pace and R. Barry, "Sparse Spatial Autoregressions",
      Statistics and Probability Letters, vol. 33, pp. 291-297,
      1997.

.. _cpu_dataset:

CPU small dataset
-----------------

This dataset measure computer systems activity by means of
(restricted) attributes and the objective is to predict
when the CPU is free in a certain portion of time.

.. topic:: References:

    * O. Okun, G. Valentini and M. Re, "Ensembles in Machine
      Learning Applications", Springer-Verlag, 2011.

.. _ecoli_dataset:

Ecoli dataset
-------------

This dataset contains protein localization sites.

.. topic:: References:

    * P. Horton and K. Nakai, "A Probablistic Classification System for
      Predicting the Cellular Localization Sites of Proteins", In Proceedings
      of the Fourth International Conference on Intelligent Systems for
      Molecular Biology, pp. 109-115, 1996.

.. _elevators_dataset:

Elevators dataset
-----------------

This dataset is obtained from the task of controlling a F16 aircraft,
and the objective is related to an action taken on the elevators of
the aircraft according to the status attributes of the aeroplane.

.. topic:: References:

    * R. Camacho, "Inducing models of human control skills using Machine
      Learning algorithms", 2000.

.. _fried_dataset:

Friedman dataset
----------------

This is an artificial dataset consisting of independent attributes which
are uniformly distributed. To obtain the value of the target variable,
the following equation is used:

.. math::

    Y = 10 \sin(\pi X_1 X_2) + 20 (X_3 - 0.5)^2 + 10 X_4 + 5 X_5 + \sigma(0, 1)

.. topic:: References:

    * J. Friedman, "Multivariate Adaptative Regression Splines",
      The Annals of Statistics, vol. 19, pp. 1-67, 1991.

    * L. Breiman, "Bagging predictors", Machine Learning,
      vol. 24, pp. 123–140, 1996.

.. _glass_dataset:

Glass identification dataset
----------------------------

This dataset was motivated by criminological investigation to study the
classification of types of glass according to their chemical properties.

.. topic:: References:

    * W. Evett and E. J. Spiehler, "Rule Induction in Forensic Science", 1987.

.. _housing_dataset:

Boston housing dataset
----------------------

This dataset contains information collected by the U.S Census
Service concerning housing in the area of Boston Mass.

.. topic:: References:

    * D. Harrison and D. L. Rubinfeld, "Hedonic prices and the demand
      for clean air", Journal of Environmental Economics and Management,
      vol. 5, pp. 81-102, 1978.

.. _iris_dataset:

Iris dataset
------------

This is perhaps the best known dataset to be found in the pattern
recognition literature. Fisher's paper is a classic in the field
and is referenced frequently to this day.

.. topic:: References:

    * R. A. Fisher, "The use of multiple measurements in taxonomic
      problems", Annual Eugenics, vol. 7, pp. 179-188, 1936.

    * R. O. Duda and P. E. Hart, "Pattern Classification and Scene
      Analysis", John Wiley & Sons, 1973.

    * B. V. Dasarathy, "Nosing Around the Neighborhood: A New System
      Structure and Classification Rule for Recognition in Partially
      Exposed Environments", IEEE Transactions on Pattern Analysis
      and Machine Intelligence, vol. 2, pp. 67-71, 1980.

    * G. W. Gates, "The Reduced Nearest Neighbor Rule", IEEE Transactions
      on Information Theory, vol. 18, pp. 431-433, 1972.

.. _letter_dataset:

Letter recognition dataset
--------------------------

This dataset contains a large number of black-and-white rectangular
pixel displays as one of the capital letters in the English alphabet.

.. topic:: References:

    * P. W. Frey and D. J. Slate, "Letter Recognition Using Holland-style
      Adaptive Classifiers", Machine Learning, vol. 6, pp. 161–182, 1991. 

.. _libras_dataset:

Libras movement dataset
-----------------------

This dataset consists of classyfing references to a hand movement type
according to a mapping operation representing the coordinates of movement.

.. topic:: References:

    * D. B. Dias, R. C. B. Madeo, T. Rocha, H. H. Bíscaro and S. M. Peres,
      "Hand Movement Recognition for Brazilian Sign Language: A Study Using
      Distance-Based Neural Networks", In Proceedings of the International
      Joint Conference on Neural Networks, pp. 697-704, 2009.

.. _pendigits_dataset:

Pen-based recognition of handwritten digits dataset
---------------------------------------------------

This dataset contains samples arising from handwritten digits characterized
by pen trajectories (successive pen points on a coordinate system).

.. topic:: References:

    * F. Alimoglu, "Combining Multiple Classifiers for Pen-Based
      Handwritten Digit Recognition", 1996. 

    * F. Alimoglu and E. Alpaydin, "Methods of Combining Multiple
      Classifiers Based on Different Representations for Pen-based
      Handwriting Recognition", In Proceedings of the Fifth Turkish
      Artificial Intelligence and Artificial Neural Networks Symposium,
      1996.

.. _satimage_dataset:

Landsat satellite dataset
-------------------------

This dataset consists of the multi-spectral values of pixels in 3x3
neighbourhoods in a satellite image, and the classification associated
with the central pixel in each neighbourhood.

.. _segment_dataset:

Image segmentation dataset
--------------------------

This dataset contains image data described by high-level attributes
of outdoor images (hand-segmented to create a classification for
every pixel).

.. _stock_dataset:

Stock prices dataset
--------------------

This dataset provides daily stock prices from January 1988 through October
1991, for 10 aerospace companies. The objective is to aproximate the price
of the 10th company given the prices of the rest. 

.. topic:: References:

    * H. Altay and I. Uysal, "Bilkent University Function Approximation
      Repository", 2000.

.. _vehicle_dataset:

Vehicle silhouettes dataset
---------------------------

This dataset purpose is to classify a given silhouette as one of
four types of vehicle, using a set of features extracted from the
silhouette (the vehicle may be viewed from one of many different
angles).

.. topic:: References:

    * J. P. Siebert, "Vehicle Recognition Using Rule Based Methods", 1987.

.. _vowel_dataset:

Vowel recognition dataset
-------------------------

This dataset consists of a three dimensional array: speaker, vowel and
input. The speakers and vowels are indexed by integers and, for each
utterance, there are floating-point input values. 

.. _wine_dataset:

Wine dataset
------------

This dataset is the result of a chemical analysis of wines grown in the
same region in Italy but derived from three different cultivars. The
analysis determined the quantities of constituents found in each of
the types of wines.

.. topic:: References:

    * S. Aeberhard, D. Coomans and O. de Vel, "Comparative analysis
      of statistical pattern recognition methods in high dimensional
      settings", Pattern Recognition, vol. 27, pp. 1065-1077, 1994.

    * S. Aeberhard, D. Coomans and O. de Vel, "Improvements to the
      classification performance of RDA", Journal of Chemometrics,
      vol. 7, pp. 99-115, 1993.

.. _wisconsin_dataset:

Breast cancer wisconsin dataset
-------------------------------

This dataset contains features computed from a digitized image of a fine
needle aspirate of a breast mass. They describe characteristics of the
cell nuclei present in the image.

.. topic:: References:

    * W. N. Street, O. L. Mangasarian and W. H. Wolberg, "An inductive
      learning approach to prognostic prediction", In Proceedings of the
      Twelfth International Conference on Machine Learning, pp. 522-530,
      1995.

    * O. L. Mangasarian, W. N. Street and W. H. Wolberg, "Breast cancer
      diagnosis and prognosis via linear programming", Operations Research,
      vol. 43, pp. 570-577, 1995.

    * W. H. Wolberg, W. N. Street, D. M. Heisey and O. L. Mangasarian,
      "Computerized breast cancer diagnosis and prognosis from fine
      needle aspirates", Archives of Surgery, vol. 130, pp. 511-516,
      1995.

    * W. H. Wolberg, W. N. Street and O. L. Mangasarian, "Image analysis
      and Machine Learning applied to breast cancer diagnosis and prognosis",
      Analytical and Quantitative Cytology and Histology, vol. 17, pp. 77-87,
      1995.

    * W. H. Wolberg, W. N. Street, D. M. Heisey and O. L. Mangasarian,
      "Computer-derived nuclear grade and breast cancer prognosis",
      Analytical and Quantitative Cytology and Histology, vol. 17,
      pp. 257-264, 1995.

.. _yeast_dataset:

Yeast dataset
-------------

This dataset consists of predicting the cellular localization sites of
proteins.

.. topic:: References:

    * P. Horton and K. Nakai, "A Probabilistic Classification System for
      Predicting the Cellular Localization Sites of Proteins", In Proceedings
      of the International Conference on Intelligent Systems for Molecular
      Biology, pp. 109-115, 1996.

.. _real_world_datasets:

Real-world datasets
===================

We provide the following functions to load real-world datasets:

.. autosummary::

    load_cold
    load_diau
    load_dtt
    load_heat
    load_spo

These datasets originate from the bioinformatics fields considering two types
of genetic data, namely phylogenetic profiles and microarray expression data
for the Yeast genome. The Yeast genome consists of genes, and each gene is
represented by an associated phylogenetic profile. Using these profiles as
input :term:`features`, the expression profile of a gene is ordered into
ranks. The use of five microarray experiments (spo, heat, dtt, cold, diau),
gives rise to five prediction problems allusing the same input features but
different target rankings.

.. topic:: References:

    * E. Hüllermeier, J. Fürnkranz, W. Cheng and K. Brinker, "Label ranking
      by learning pairwise preferences", Artificial Intelligence, vol. 172,
      pp. 1897–1916, 2008.

.. References

.. _UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
