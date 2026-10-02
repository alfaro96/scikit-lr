import importlib
import inspect
import pkgutil

import pytest

import sklr

# numpydoc is a dependency of the documentation, so the jobs that only test the
# installed package skip these tests, which run with the development environment
numpydoc_validation = pytest.importorskip("numpydoc.validate")

# The checks of numpydoc that scikit-learn ignores: the layout of the quotes
# (GL01, GL02) and the first line of the Returns section, which may name the
# returned value (RT02), everywhere, and the extended summary, the See Also
# section and the examples (ES01, SA01, EX01) in the methods, which are
# documented in the page of their class
IGNORED_CHECKS = {"GL01", "GL02", "RT02"}
IGNORED_METHOD_CHECKS = IGNORED_CHECKS | {"ES01", "SA01", "EX01"}


def _public_modules(module=sklr):
    """Yield `module` and its public submodules, recursively."""
    yield module
    # Only a package has a __path__ in which to look for submodules
    path = getattr(module, "__path__", [])
    for module_info in pkgutil.iter_modules(path, f"{module.__name__}."):
        name = module_info.name.rpartition(".")[2]
        if not name.startswith("_") and name not in {"conftest", "tests"}:
            yield from _public_modules(importlib.import_module(module_info.name))


def _public_objects():
    """Yield each public function, class and method, with the checks to ignore."""
    # The public names are often defined in private modules, so an object is
    # validated once, under the first public module that exports it
    paths = {}
    for module in _public_modules():
        for name, obj in inspect.getmembers(module):
            if (
                not name.startswith("_")
                and (inspect.isclass(obj) or inspect.isroutine(obj))
                and obj.__module__.partition(".")[0] == "sklr"
            ):
                paths.setdefault(obj, f"{module.__name__}.{name}")

    for obj, path in paths.items():
        yield path, IGNORED_CHECKS
        if inspect.isclass(obj):
            for name in dir(obj):
                if name.startswith("_"):
                    continue
                attr = getattr(obj, name)
                if callable(attr) or isinstance(attr, property):
                    yield f"{path}.{name}", IGNORED_METHOD_CHECKS


@pytest.mark.parametrize(
    "path, ignored_checks",
    [pytest.param(path, checks, id=path) for path, checks in _public_objects()],
)
def test_docstring(path, ignored_checks):
    result = numpydoc_validation.validate(path)
    errors = [
        f"{code}: {message}"
        for code, message in result["errors"]
        if code not in ignored_checks
    ]
    assert not errors, "\n".join([result["file"], *errors])
