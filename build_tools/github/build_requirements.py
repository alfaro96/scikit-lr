"""Print the build requirements of the package, except the given packages.

The package is then built without isolation against the given packages
installed apart, which may be releases that its build requirements do not
allow yet.

Run it from the root of the repository, with the names of the packages to
leave out as arguments.
"""

import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

excluded = {canonicalize_name(name) for name in sys.argv[1:]}

with open("pyproject.toml", "rb") as file:
    requires = tomllib.load(file)["build-system"]["requires"]

for requirement in map(Requirement, requires):
    if canonicalize_name(requirement.name) not in excluded:
        print(requirement)
