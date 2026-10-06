.. _api_ref:

API reference
=============

This is the class and function reference of scikit-lr. Each entry links to a
page with its full description, and the :ref:`user guide <user_guide>` gives
more details on how to use them.

:mod:`sklr`: Settings and information tools
-------------------------------------------

.. module:: sklr

.. autosummary::
   :toctree: ../modules/generated/
   :template: base.rst

   show_versions

:mod:`sklr.base`: Base classes and utility functions
----------------------------------------------------

.. module:: sklr.base

.. autosummary::
   :toctree: ../modules/generated/
   :template: base.rst

   LabelRankerMixin
   PartialLabelRankerMixin
   is_label_ranker
   is_partial_label_ranker

:mod:`sklr.metrics`: Metrics
----------------------------

.. module:: sklr.metrics

See the :ref:`model_evaluation` section of the user guide for further details.

Scoring interface
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../modules/generated/
   :template: base.rst

   get_scorer
   get_scorer_names

Label ranking metrics
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../modules/generated/
   :template: base.rst

   kendall_distance
   kendall_tau_score

Partial label ranking metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../modules/generated/
   :template: base.rst

   kendall_tau_x_score

:mod:`sklr.utils`: Utilities
----------------------------

.. module:: sklr.utils

.. autosummary::
   :toctree: ../modules/generated/
   :template: base.rst

   check_ranking
   type_of_ranking
