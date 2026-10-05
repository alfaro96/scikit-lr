"""Private API of scikit-learn used by scikit-lr.

The private API of scikit-learn may change without deprecation, so the rest of
:mod:`sklr` imports it only from this module, and the ``no-private-sklearn-imports``
pre-commit hook enforces it. scikit-lr is built and run against a single minor
release of scikit-learn, so each name is imported here as that release defines it.
When moving to a new minor release, the names that have changed are adapted here,
keeping the interface that the rest of the package uses.
"""
