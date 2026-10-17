"""Check that the given Python files do not import private scikit-learn names.

A name is private when any part of its dotted path starts with an underscore,
other than the dunder names such as ``__version__``, whether it is imported as a
module (``import sklearn.utils._param_validation``) or from one
(``from sklearn.utils import _safe_indexing``). The private API of scikit-learn
may change without deprecation, so it is only imported from
``sklr/utils/_sklearn_compat.py``, which ``pre-commit`` leaves out of this check.

The files are parsed instead of searched line by line, so the imports split
over several lines are found too.
"""

import ast
import sys


def is_private(dotted_name):
    """Return whether any part of `dotted_name` is private."""
    return any(
        part.startswith("_") and not (part.startswith("__") and part.endswith("__"))
        for part in dotted_name.split(".")
    )


def private_imports(source):
    """Yield the line and the name of each private scikit-learn import."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "sklearn" and is_private(alias.name):
                    yield node.lineno, alias.name
        # Relative imports have no module or a level above zero
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] != "sklearn":
                continue
            if is_private(node.module):
                yield node.lineno, node.module
                continue
            for alias in node.names:
                if is_private(alias.name):
                    yield node.lineno, f"{node.module}.{alias.name}"


def main(paths):
    errors = 0
    for path in paths:
        with open(path, encoding="utf-8") as file:
            source = file.read()
        for lineno, name in private_imports(source):
            print(
                f"{path}:{lineno}: imports {name}, private API of scikit-learn, "
                "which is only imported from sklr/utils/_sklearn_compat.py"
            )
            errors += 1
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
