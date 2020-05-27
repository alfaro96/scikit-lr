==========================
Frequently Asked Questions
==========================

Here we try to give some answers to questions that regularly pop-up.

Why scikit?
===========

SciKits (short for SciPy Toolkits) are add-on packages for SciPy,
hosted and developed separately from the main SciPy distribution.
You can find a list at: https://scikits.appspot.com/scikits.

How can I install scikit-lr?
============================

See :ref:`installation`.

.. _help_with_usage:

What is the best way to get help on scikit-lr usage?
====================================================

Please, make sure to include a minimal reproduction code snippet (ideally,
shorter than 10 lines) that highlights your problem on a toy dataset (for
instance, from :mod:`sklr.datasets` or randomly generated with functions
of :mod:`numpy.random` with a fixed random seed). Remove any line of code
that is not necessary to reproduce your problem.

The problem should be reproducible by simply *copy-pasting* your code
snippet in a Python shell with scikit-lr installed. Do not forget to
include the import statements. More guidance to write good reproduction
code snippets can be found at: https://stackoverflow.com/help/mcve.

If your problem raises an exception that you do not understand, make sure
to include the full traceback that you obtain when running the reproduction
script.

For bug reports, make use of the `issue tracker`_.

**Do not email any authors directly to ask for assistance,
report bugs, or for any other issue related to scikit-lr.**

How do I set a ``random_state`` for an entire execution?
========================================================

For testing and reproducibility, it is important to have the entire execution
controlled by a single seed for the pseudo-random number generator used in
algorithms that have a randomized component. Scikit-lr does not use its own
global random state, but whenever a :class:`numpy.random.RandomState` instance
or an integer random seed is not provided as an argument, it relies on the
NumPy global random state, which can be set using :func:`numpy.random.seed`.

For example, to set a NumPy global random state to 42, one could execute the
following in their script::

    import numpy as np
    np.random.seed(42)

However, a global random state is prone to modification by other
code during execution. Thus, the only way to ensure reproducibility
is to pass :class:`numpy.random.RandomState` instances everywhere,
and ensure that both, estimators and cross-validation splitters have
their ``random_state`` parameter set.

.. References

.. _issue tracker: https://github.com/alfaro96/scikit-lr/issues
