"""Print the minimum supported version of every runtime dependency.

The versions are the lower bounds of the ranges in ``pyproject.toml``,
written as requirements that pin them, so they can be installed to test
the package with its oldest supported dependencies.

Run it from the root of the repository.
"""

import sys
import tomllib

from packaging.requirements import Requirement

with open("pyproject.toml", "rb") as file:
    dependencies = tomllib.load(file)["project"]["dependencies"]

for requirement in map(Requirement, dependencies):
    minimum = [
        specifier.version
        for specifier in requirement.specifier
        if specifier.operator in (">=", "~=")
    ]
    if len(minimum) != 1:
        sys.exit(
            f"{requirement.name} in project.dependencies in pyproject.toml must "
            "have a single '>=' or '~=' bound to find its minimum supported "
            f"version, got {str(requirement.specifier)!r}"
        )
    print(f"{requirement.name}=={minimum[0]}")
