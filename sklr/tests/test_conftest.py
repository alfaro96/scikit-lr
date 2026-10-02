"""Check that the doctests run without the globals of their module.

The example below would see ``MODULE_GLOBAL`` if ``sklr/conftest.py`` did not
empty the globals of every doctest.

>>> "MODULE_GLOBAL" in globals()
False
"""

MODULE_GLOBAL = None
