.. _glossary:

.. currentmodule:: sklr

========
Glossary
========

This glossary aims to describe the concepts and either detail their
corresponding API, or link to other relevant parts of the documentation
which do so. By linking to glossary entries from the :ref:`api_reference`
and :ref:`user_guide`, we may minimize redundancy and inconsistency.

Most of the concepts are available in the `glossary of scikit-learn`_,
but more specific sets of related terms are listed below:

General concepts
================

.. glossary::
    ranking
    rankings
    partial ranking
    partial rankings
        An order of preference of (some) discrete values defined over a
        finite set. Rankings may be *complete* (all values are ranked)
        or *incomplete* (only some values are ranked). Rankings may also
        be classified as *with-ties* (partial rankings) or *without-ties*
        (rankings) depending on if they present lack of preference
        information among some of the ranked values.

Class APIs and estimator types
==============================

.. glossary::

    label ranker
    label rankers
        A :term:`supervised` :term:`predictor` with a :term:`ranking`
        defined over a finite set of discrete values.

        Label rankers usually inherit from :class:`base.LabelRankerMixin`,
        which sets their :term:`_estimator_type` attribute.

        A label ranker can be distinguised from other
        estimators with :func:`~base.is_label_ranker`.

        A label ranker must implement: :term:`fit`, :term:`predict` and
        :term:`score`.

    partial label ranker
    partial label rankers
        A :term:`supervised` :term:`predictor` with a
        :term:`partial ranking` defined over a finite
        set of discrete values.

        Partial label rankers usually inherit from
        :class:`base.PartialLabelRankerMixin`, which
        sets their :term:`_estimator_type` attribute.

        A partial label ranker can be distinguised from other
        estimators with :func:`~base.is_partial_label_ranker`.

        A partial label ranker must implement: :term:`fit`,
        :term:`predict` and :term:`score`.

Target types
============

.. glossary::

    label_ranking
        A label ranking problem where each :term:`sample`'s
        :term:`target` is a :term:`ranking` defined over a
        finite set of discrete values.

    partial_label_ranking
        A partial label ranking problem where each :term:`sample`'s
        :term:`target` is a :term:`partial ranking` defined over a
        finite set of discrete values.

.. References

.. _glossary of scikit-learn: https://scikit-learn.org/stable/glossary.html
