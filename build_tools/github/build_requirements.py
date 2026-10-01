"""Print the build requirements of the package, except ``scikit-learn``.

The package is then built without isolation against a ``scikit-learn`` installed
apart, which may be a release that its build requirements do not allow yet.

Run it from the root of the repository.
"""

import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

with open("pyproject.toml", "rb") as file:
    requires = tomllib.load(file)["build-system"]["requires"]

for requirement in map(Requirement, requires):
    if canonicalize_name(requirement.name) != "scikit-learn":
        print(requirement)
