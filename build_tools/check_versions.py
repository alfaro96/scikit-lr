"""Check that the versions declared in more than one file agree.

The versions of some packages, and the supported Python versions, are declared
in more than one file, so they are checked against each other to keep them from
drifting apart.

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

The minimum supported Python version is the lower bound of ``requires-python``
in ``pyproject.toml``, and the classifiers list the supported versions from it.
The unit tests must run on all the versions in the classifiers, the job with
the minimum dependencies on the minimum version, and the workflows that set up
a single Python version, instead of one from a matrix, must use the minimum.

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

WORKFLOWS_DIR = Path(".github/workflows")

# Workflow whose matrix runs the tests on every supported Python version,
# and the path to that matrix in the workflow
UNIT_TESTS_WORKFLOW = WORKFLOWS_DIR / "unit-tests.yml"
UNIT_TESTS_MATRIX = ["jobs", "unit-tests", "strategy", "matrix"]

# The classifiers of the supported Python versions
PYTHON_CLASSIFIER = re.compile(
    r"^Programming Language :: Python :: (?P<version>3\.\d+)$"
)

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


def read_workflows():
    """Return the parsed GitHub Actions workflows, by path."""
    return {
        path: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
    }


def lookup(data, keys):
    """Return the value at the path of `keys` in nested mappings, or None."""
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def minimum_python(requires_python):
    """Return the minimum supported Python version, or None if not found."""
    minimum = [
        specifier.version
        for specifier in SpecifierSet(requires_python)
        if specifier.operator == ">="
    ]
    return minimum[0] if len(minimum) == 1 else None


def check_python(environment, minimum):
    """Check that the environment uses the minimum supported Python version."""
    expected = f"={minimum}"
    spec = environment.get("python")
    if spec != expected:
        message = (
            f"python in environment.yml must be {expected!r}, the minimum "
            f"supported version in requires-python in pyproject.toml, got {spec!r}"
        )
        return [message]
    return []


def check_classifiers(versions, minimum):
    """Check that the Python classifiers start at the minimum supported version."""
    oldest = min(versions, key=lambda version: tuple(map(int, version.split("."))))
    if oldest != minimum:
        message = (
            f"the oldest Python version in the classifiers in pyproject.toml must "
            f"be {minimum!r}, the minimum supported version in requires-python, "
            f"got {oldest!r}"
        )
        return [message]
    return []


def check_unit_tests(matrix, versions, minimum):
    """Check the Python versions of the unit tests matrix."""
    if not isinstance(matrix, dict):
        message = (
            f"{UNIT_TESTS_WORKFLOW} must have a {'.'.join(UNIT_TESTS_MATRIX)} "
            "mapping, which runs the tests on the supported Python versions"
        )
        return [message]
    errors = []
    python = matrix.get("python")
    if sorted(map(str, python or [])) != sorted(versions):
        errors.append(
            f"the Python versions of the matrix in {UNIT_TESTS_WORKFLOW} must be "
            f"{versions!r}, as in the classifiers in pyproject.toml, got {python!r}"
        )
    for include in matrix.get("include", []):
        if include.get("dependencies") == "minimum":
            python = include.get("python")
            if str(python) != minimum:
                errors.append(
                    "the job with the minimum dependencies in "
                    f"{UNIT_TESTS_WORKFLOW} must use Python {minimum!r}, the "
                    "minimum supported version in requires-python in "
                    f"pyproject.toml, got {python!r}"
                )
    return errors


def check_setup_python(workflows, minimum):
    """Check that the workflows set up the minimum supported Python version.

    The versions taken from a matrix, written as an expression, are checked
    along with the matrix instead.
    """
    errors = []
    for path, workflow in workflows.items():
        for job_id, job in (workflow.get("jobs") or {}).items():
            for step in job.get("steps", []):
                if not str(step.get("uses", "")).startswith("actions/setup-python@"):
                    continue
                python = lookup(step, ["with", "python-version"])
                if "${{" in str(python):
                    continue
                if str(python) != minimum:
                    errors.append(
                        f"the job {job_id} in {path} must set up Python "
                        f"{minimum!r}, the minimum supported version in "
                        f"requires-python in pyproject.toml, got {python!r}"
                    )
    return errors


def check_python_versions(environment, project, workflows):
    """Check every declaration of the supported Python versions."""
    requires_python = project["requires-python"]
    minimum = minimum_python(requires_python)
    if minimum is None:
        message = (
            "requires-python in pyproject.toml must have a single '>=' bound to "
            f"find the minimum supported version, got {requires_python!r}"
        )
        return [message]
    versions = [
        match["version"]
        for match in map(PYTHON_CLASSIFIER.match, project["classifiers"])
        if match is not None
    ]
    if not versions:
        message = (
            "the classifiers in pyproject.toml must list the supported Python "
            "versions, such as 'Programming Language :: Python :: "
            f"{minimum}', got none"
        )
        return [message]
    matrix = lookup(workflows.get(UNIT_TESTS_WORKFLOW), UNIT_TESTS_MATRIX)
    return [
        *check_python(environment, minimum),
        *check_classifiers(versions, minimum),
        *check_unit_tests(matrix, versions, minimum),
        *check_setup_python(workflows, minimum),
    ]


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
                f"{name} must have the same range in project.dependencies and in "
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
    workflows = read_workflows()

    runtime = read_requirements(pyproject["project"]["dependencies"])
    build = read_requirements(pyproject["build-system"]["requires"])
    # The environment has the runtime range of the packages needed both
    # at runtime and at build time, which is the one checked for them
    build_only = {name: build[name] for name in build if name not in runtime}

    errors = [
        *check_cimported(runtime, build),
        *check_python_versions(environment, pyproject["project"], workflows),
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
