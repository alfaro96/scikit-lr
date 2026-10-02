"""Check that the distributions to publish are the ones of the release.

A release is published from a tag, which must be the version of the package,
so the files on PyPI match the tag that they were built from. The directory of
the distributions must have the source distribution and a wheel for every
platform in the matrix of the workflow and every Python version selected in
``[tool.cibuildwheel]``, all of them of that version. A missing wheel would
make the users of its platform compile the package from the source
distribution, and a wrong file cannot be fixed once it is on PyPI, which never
accepts a file name twice, even after the file is deleted.

Run it from the root of the repository with the directory of the
distributions, in a GitHub Actions job started from the tag.
"""

import os
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import yaml
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)

DISTRIBUTION = "scikit-lr"

# Workflow whose matrix builds the wheels on every platform,
# and the path to that matrix in the workflow
WHEELS_WORKFLOW = Path(".github/workflows/wheels.yml")
WHEELS_MATRIX = ["jobs", "wheels", "strategy", "matrix", "include"]


def package_version():
    """Return the version of the package."""
    return subprocess.run(
        [sys.executable, "sklr/_build_utils/version.py"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def expected_wheels():
    """Return the number of wheels expected for each Python version."""
    platforms = yaml.safe_load(WHEELS_WORKFLOW.read_text(encoding="utf-8"))
    for key in WHEELS_MATRIX:
        platforms = platforms[key]
    with open("pyproject.toml", "rb") as file:
        build = tomllib.load(file)["tool"]["cibuildwheel"]["build"]
    # The patterns select CPython versions, and the wheels have the
    # same interpreter tag before the "-"
    return {pattern.split("-")[0]: len(platforms) for pattern in build}


def check_tag(version):
    """Check that the job runs from a tag that is the version of the package."""
    ref_type = os.environ["GITHUB_REF_TYPE"]
    ref_name = os.environ["GITHUB_REF_NAME"]
    if ref_type != "tag":
        message = (
            f"a release must be published from a tag, got the {ref_type} {ref_name!r}"
        )
        return [message]
    if ref_name != version:
        message = (
            f"the tag must be the version of the package, {version!r} in "
            f"sklr/__init__.py, got {ref_name!r}"
        )
        return [message]
    return []


def check_distributions(paths, version):
    """Check the names and the versions of the distributions in `paths`."""
    errors = []
    sdists = []
    wheels = Counter()
    for path in paths:
        try:
            if path.name.endswith(".tar.gz"):
                name, found_version = parse_sdist_filename(path.name)
                sdists.append(path)
            else:
                name, found_version, _, tags = parse_wheel_filename(path.name)
                wheels.update({tag.interpreter for tag in tags})
        except (InvalidSdistFilename, InvalidWheelFilename) as error:
            errors.append(f"{path}: not a distribution, {error}")
            continue
        if name != canonicalize_name(DISTRIBUTION) or str(found_version) != version:
            errors.append(
                f"{path}: expected a distribution of {DISTRIBUTION} {version}, "
                f"got {name} {found_version}"
            )
    if len(sdists) != 1:
        errors.append(f"expected a source distribution, got {len(sdists)}")
    expected = expected_wheels()
    for interpreter in sorted(expected.keys() | wheels.keys()):
        if wheels[interpreter] != expected.get(interpreter, 0):
            errors.append(
                f"expected {expected.get(interpreter, 0)} wheel(s) for "
                f"{interpreter}, one for each platform in {WHEELS_WORKFLOW}, "
                f"got {wheels[interpreter]}"
            )
    return errors


def main(dist_dir):
    paths = sorted(path for path in Path(dist_dir).iterdir() if path.is_file())
    version = package_version()
    errors = [*check_tag(version), *check_distributions(paths, version)]
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print(f"Publishing {DISTRIBUTION} {version}:")
        print("\n".join(path.name for path in paths))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
