import pytest


def pytest_collection_modifyitems(config, items):
    # A doctest runs with the globals of its module, so an example could use a
    # name that it does not import and still pass. Empty them, so every example
    # works as is when copied
    for item in items:
        if isinstance(item, pytest.DoctestItem):
            item.dtest.globs = {}
