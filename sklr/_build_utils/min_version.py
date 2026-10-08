"""Print the minimum version of a build requirement from ``pyproject.toml``.

The minimum is the version of the ``>=`` specifier of the requirement whose name
is given as the only argument, so ``meson.build`` can check that the installed
version meets it when building without isolation, where ``pip`` does not check it
by default.
"""

import re
import sys
import tomllib
from pathlib import Path

pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"


def normalize(name):
    """Normalize a distribution name as in PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def min_version(name):
    """Return the minimum version of the build requirement called `name`."""
    with pyproject.open("rb") as file:
        requires = tomllib.load(file)["build-system"]["requires"]

    # Parse the requirements by hand, since the build may run without packaging
    for requirement in requires:
        # The name, the optional extras and the specifiers up to the marker
        match = re.match(
            r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*([^;]*)", requirement
        )
        if match and normalize(match[1]) == normalize(name):
            for specifier in match[2].split(","):
                if specifier.strip().startswith(">="):
                    return specifier.strip().removeprefix(">=").strip()
            raise SystemExit(f"No minimum version for {requirement!r} in {pyproject}")

    raise SystemExit(f"No build requirement named {name!r} in {pyproject}")


# pytest imports this module to collect its doctests
if __name__ == "__main__":
    print(min_version(sys.argv[1]))
