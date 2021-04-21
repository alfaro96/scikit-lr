.. _api_reference:

=============
API Reference
=============

This is the class and function reference of scikit-lr. Please, refer to the
:ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on
their uses. For reference on concepts repeated across the API, see the
:ref:`glossary <glossary>`.

:mod:`sklr.base`: Base classes and utility functions
====================================================

.. automodule:: sklr.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: sklr

Base classes
------------

.. autosummary::
    :nosignatures:
    :template: class.rst
    :toctree: generated/

    base.LabelRankerMixin
    base.PartialLabelRankerMixin
    base.TransformerMixin

Functions
---------

.. autosummary::
    :toctree: generated/

    base.is_label_ranker
    base.is_partial_label_ranker

:mod:`sklr.datasets`: Datasets
==============================

.. automodule:: sklr.datasets
    :no-members:
    :no-inherited-members:

**User guide:** See the :ref:`datasets` section for further details.

.. currentmodule:: sklr

Loaders
-------

.. autosummary::
    :toctree: generated/

    datasets.load_authorship
    datasets.load_blocks
    datasets.load_bodyfat
    datasets.load_breast
    datasets.load_calhousing
    datasets.load_cold
    datasets.load_cpu
    datasets.load_diau
    datasets.load_dtt
    datasets.load_ecoli
    datasets.load_elevators
    datasets.load_fried
    datasets.load_glass
    datasets.load_heat
    datasets.load_housing
    datasets.load_iris
    datasets.load_letter
    datasets.load_libras
    datasets.load_pendigits
    datasets.load_satimage
    datasets.load_segment
    datasets.load_spo
    datasets.load_stock
    datasets.load_vehicle
    datasets.load_vowel
    datasets.load_wine
    datasets.load_wisconsin
    datasets.load_yeast
