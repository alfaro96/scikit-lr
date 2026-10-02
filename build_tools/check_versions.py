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

The tools pinned in ``environment.yml`` that also run as ``pre-commit`` hooks must
have the version of their hook in ``.pre-commit-config.yaml``. The packages
installed in the environments of the hooks must be pinned, so the result of
the hooks only changes with that file, and the pinned versions of the runtime
dependencies must be within their ranges in ``pyproject.toml``. Renovate updates
those pins, so its configuration must restrict these dependencies to the same
ranges, and every hook with packages installed in its environment must declare
its language, since Renovate leaves out the packages of the hooks that do not.

The minimum supported Python version is the lower bound of ``requires-python``
in ``pyproject.toml``, and the classifiers list the supported versions from it.
The unit tests must run on all the versions in the classifiers, the job with
the minimum dependencies on the minimum version, and the workflows that set up
a single Python version, instead of one from a matrix, must use the minimum.
The wheels must be built for the versions in the classifiers too, because
``cibuildwheel`` would otherwise build them for every version that it knows of,
including the ones released after the classifiers were last updated.

Run it from the root of the repository.
"""

import json
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

RENOVATE_CONFIG = Path(".github/renovate.json")

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
    """Return the value at the path of `keys` in nested mappings, or ``None``."""
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def minimum_python(requires_python):
    """Return the minimum supported Python version, or ``None`` if not found."""
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


def check_cibuildwheel(build, versions):
    """Check that the wheels are built for the supported Python versions."""
    # The wheels can also be built for free-threaded Python (without the GIL),
    # whose names have a "t" after the version, so the patterns end the version
    # with "-" to leave those wheels out
    expected = sorted(f"cp{version.replace('.', '')}-*" for version in versions)
    if not isinstance(build, list) or sorted(build) != expected:
        message = (
            "build in [tool.cibuildwheel] in pyproject.toml must be "
            f"{expected!r}, the Python versions in the classifiers, got {build!r}"
        )
        return [message]
    return []


def check_python_versions(environment, pyproject, workflows):
    """Check every declaration of the supported Python versions."""
    project = pyproject["project"]
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
    build = lookup(pyproject, ["tool", "cibuildwheel", "build"])
    return [
        *check_python(environment, minimum),
        *check_classifiers(versions, minimum),
        *check_unit_tests(matrix, versions, minimum),
        *check_setup_python(workflows, minimum),
        *check_cibuildwheel(build, versions),
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
    """Check that the packages installed in the hooks are pinned and supported."""
    errors = []
    for repo in repos:
        for hook in repo["hooks"]:
            source = (
                f"the additional_dependencies of the {hook['id']} hook in "
                ".pre-commit-config.yaml"
            )
            for dependency in hook.get("additional_dependencies", []):
                requirement = Requirement(dependency)
                pinned = [
                    specifier.version
                    for specifier in requirement.specifier
                    if specifier.operator == "=="
                ]
                if len(requirement.specifier) != 1 or len(pinned) != 1:
                    errors.append(
                        f"{requirement.name} must be pinned with '==' in {source}, "
                        f"got {dependency!r}"
                    )
                    continue
                expected = runtime.get(canonicalize_name(requirement.name))
                if expected is not None and pinned[0] not in expected:
                    errors.append(
                        f"{requirement.name} in {source} must be within the range "
                        f"{str(expected)!r} of project.dependencies in "
                        f"pyproject.toml, got {pinned[0]!r}"
                    )
    return errors


def check_hook_languages(repos):
    """Check that the hooks with packages installed in them declare their language."""
    errors = []
    for repo in repos:
        for hook in repo["hooks"]:
            if hook.get("additional_dependencies") and "language" not in hook:
                errors.append(
                    f"the {hook['id']} hook in .pre-commit-config.yaml must declare "
                    "its language, because Renovate only updates the "
                    "additional_dependencies of the hooks that declare it"
                )
    return errors


def check_renovate(renovate, repos, runtime):
    """Check that Renovate keeps the runtime dependencies of the hooks in range."""
    allowed = {
        canonicalize_name(name): rule["allowedVersions"]
        for rule in renovate.get("packageRules", [])
        if "allowedVersions" in rule
        for name in rule.get("matchPackageNames", [])
    }
    names = {
        canonicalize_name(Requirement(dependency).name)
        for repo in repos
        for hook in repo["hooks"]
        for dependency in hook.get("additional_dependencies", [])
    }
    errors = []
    for name in sorted(names & runtime.keys()):
        expected = runtime[name]
        try:
            spec = SpecifierSet(allowed[name])
        except (KeyError, InvalidSpecifier):
            spec = None
        if spec != expected:
            errors.append(
                f"{name} must have the allowedVersions {str(expected)!r} in "
                f"{RENOVATE_CONFIG}, its range in project.dependencies in "
                f"pyproject.toml, got {allowed.get(name)!r}"
            )
    return errors


def main():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    environment = read_environment(Path("environment.yml"))
    pre_commit = yaml.safe_load(
        Path(".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    workflows = read_workflows()
    renovate = json.loads(RENOVATE_CONFIG.read_text(encoding="utf-8"))

    runtime = read_requirements(pyproject["project"]["dependencies"])
    build = read_requirements(pyproject["build-system"]["requires"])
    # The environment has the runtime range of the packages needed both
    # at runtime and at build time, which is the one checked for them
    build_only = {name: build[name] for name in build if name not in runtime}

    errors = [
        *check_cimported(runtime, build),
        *check_python_versions(environment, pyproject, workflows),
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
        *check_hook_languages(pre_commit["repos"]),
        *check_renovate(renovate, pre_commit["repos"], runtime),
    ]
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
