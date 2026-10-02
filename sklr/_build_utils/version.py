#!/usr/bin/env python3
"""Extract the version number from ``sklr/__init__.py``."""

import ast
from pathlib import Path

sklr_init = Path(__file__).parent.parent / "__init__.py"


def version():
    """Return the value assigned to ``__version__`` in ``sklr/__init__.py``."""
    # Parse the module instead of importing it, because the package
    # cannot be imported before its extension modules are built
    for node in ast.parse(sklr_init.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit(f"No __version__ assignment found in {sklr_init}")


# pytest imports this module to collect its doctests
if __name__ == "__main__":
    print(version())
