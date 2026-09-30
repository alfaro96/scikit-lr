"""Check that the development environment agrees with the package metadata.

The versions of some packages are declared in more than one file, so they are
checked against each other to keep them from drifting apart.

In ``pyproject.toml``, the packages whose ``.pxd`` files are cimported must
have the same range in the runtime and in the build dependencies, because the
extension modules only work with the minor release they were built against.
The runtime and build dependencies in ``environment.yml`` must have the same
ranges as in ``pyproject.toml``, and its Python version must be the minimum
supported one.

The tools pinned in ``environment.yml`` that also run as pre-commit hooks must
have the version of their hook in ``.pre-commit-config.yaml``, and the runtime
dependencies installed in the environments of the hooks must have the same
ranges as in ``pyproject.toml``.

Run it from the root of the repository.
"""

import re
import sys
import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name

# Repositories of the pre-commit hooks that run a tool also pinned
# in environment.yml, mapped to the name of the conda package
HOOK_PACKAGES = {
    "https://github.com/astral-sh/ruff-pre-commit": "ruff",
    "https://github.com/facebook/pyrefly-pre-commit": "pyrefly",
}

# Packages whose .pxd files are cimported by the extension modules, which
# must run with the same minor release they were built against
CIMPORTED_PACKAGES = ["scikit-learn"]

# A conda match specification as written in environment.yml: the package name
# followed by an optional version specification, and an optional comment
CONDA_SPEC = re.compile(r"^\s*(?P<name>[A-Za-z0-9_.\-]+)\s*(?P<spec>[^#\s]*)")


def read_environment(path):
    """Return the version specification of each conda package, by name."""
    environment = yaml.safe_load(path.read_text(encoding="utf-8"))
    packages = {}
    for dependency in environment["dependencies"]:
        # Skip the nested "pip:" sections, which are not conda packages
        if isinstance(dependency, str):
            match = CONDA_SPEC.match(dependency)
            packages[canonicalize_name(match["name"])] = match["spec"]
    return packages


def read_requirements(requirements):
    """Return the specifier set of each requirement, by name."""
    return {
        canonicalize_name(requirement.name): requirement.specifier
        for requirement in map(Requirement, requirements)
    }


def check_python(environment, requires_python):
    """Check that the environment uses the minimum supported Python version."""
    minimum = [
        specifier.version
        for specifier in SpecifierSet(requires_python)
        if specifier.operator == ">="
    ]
    if len(minimum) != 1:
        message = (
            "requires-python in pyproject.toml must have a single '>=' bound to "
            f"find the minimum supported version, got {requires_python!r}"
        )
        return [message]
    expected = f"={minimum[0]}"
    spec = environment.get("python")
    if spec != expected:
        message = (
            "python in environment.yml must be the minimum supported version "
            f"{expected!r}, as in requires-python {requires_python!r} in "
            f"pyproject.toml, got {spec!r}"
        )
        return [message]
    return []


def check_dependencies(source, specs, dependencies, table, required):
    """Check that the packages in `specs` have the ranges in `dependencies`.

    If `required`, every dependency must also be in `specs`.
    """
    errors = []
    for name, expected in dependencies.items():
        if name not in specs:
            if required:
                errors.append(
                    f"{name} is in {table} in pyproject.toml, but not in {source}"
                )
            continue
        try:
            spec = SpecifierSet(specs[name])
        except InvalidSpecifier:
            spec = None
        if spec != expected:
            errors.append(
                f"{name} must have the range {str(expected)!r} in {source}, as in "
                f"{table} in pyproject.toml, got {specs[name]!r}"
            )
    return errors


def check_cimported(runtime, build):
    """Check that the cimported packages have the same runtime and build range."""
    errors = []
    for name in CIMPORTED_PACKAGES:
        if runtime.get(name) != build.get(name):
            errors.append(
                "{name} must have the same range in project.dependencies and in "
                "build-system.requires in pyproject.toml, because its .pxd files "
                f"are cimported, got {str(runtime.get(name))!r} and "
                f"{str(build.get(name))!r}"
            )
    return errors


def check_hook_versions(environment, repos):
    """Check that the tools in the environment have the version of their hook."""
    errors = []
    for repo in repos:
        name = HOOK_PACKAGES.get(repo["repo"])
        if name is None:
            continue
        expected = f"={repo['rev'].removeprefix('v')}"
        spec = environment.get(name)
        if spec != expected:
            errors.append(
                f"{name} in environment.yml must be {expected!r}, the version of "
                f"its hook {repo['repo']} in .pre-commit-config.yaml, got {spec!r}"
            )
    return errors


def check_hook_dependencies(repos, runtime):
    """Check the ranges of the runtime dependencies installed in the hooks."""
    errors = []
    for repo in repos:
        for hook in repo["hooks"]:
            specs = {}
            for dependency in hook.get("additional_dependencies", []):
                requirement = Requirement(dependency)
                specs[canonicalize_name(requirement.name)] = str(requirement.specifier)
            errors += check_dependencies(
                f"the additional_dependencies of the {hook['id']} hook in "
                ".pre-commit-config.yaml",
                specs,
                runtime,
                "project.dependencies",
                required=False,
            )
    return errors


def main():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    environment = read_environment(Path("environment.yml"))
    pre_commit = yaml.safe_load(
        Path(".pre-commit-config.yaml").read_text(encoding="utf-8")
    )

    runtime = read_requirements(pyproject["project"]["dependencies"])
    build = read_requirements(pyproject["build-system"]["requires"])
    # The environment has the runtime range of the packages needed both
    # at runtime and at build time, which is the one checked for them
    build_only = {name: build[name] for name in build if name not in runtime}

    errors = [
        *check_cimported(runtime, build),
        *check_python(environment, pyproject["project"]["requires-python"]),
        *check_dependencies(
            "environment.yml",
            environment,
            runtime,
            "project.dependencies",
            required=True,
        ),
        *check_dependencies(
            "environment.yml",
            environment,
            build_only,
            "build-system.requires",
            required=True,
        ),
        *check_hook_versions(environment, pre_commit["repos"]),
        *check_hook_dependencies(pre_commit["repos"], runtime),
    ]
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
