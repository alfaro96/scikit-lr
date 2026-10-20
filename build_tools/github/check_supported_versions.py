"""Report the changes to the supported versions that are due.

scikit-lr follows SPEC 0 (https://scientific-python.org/specs/spec-0000/): each
minor release of Python is supported for three years after its first release, and
each minor release of NumPy and SciPy for two years, but never below the versions
that the supported scikit-learn requires. Its minimum supported versions are
raised in the development branch before the first release of each minor release,
so the report lists the ones that SPEC 0 allows raising by then, which is today
unless ``--date`` gives the planned date of the release.

scikit-lr is also built against a single minor release of scikit-learn, so a new
minor release of scikit-learn opens a new release line of scikit-lr, and a new
Python version can only be supported once NumPy, SciPy and the supported
scikit-learn publish wheels for it, since scikit-lr is built and tested against
them. The report lists both too.

The files of each release, with their upload dates, come from the Index API of
PyPI, the requirements of scikit-learn from its JSON API, and the release dates of
Python from python.org. The report is a Markdown list, empty when there is nothing
to change.

Run it from the root of the repository.
"""

import argparse
import datetime
import json
import re
import tomllib
import urllib.request
from collections import defaultdict
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version

# Years of support of each minor release after its first release, by package
SUPPORT_YEARS = {"python": 3, "numpy": 2, "scipy": 2}

NAMES = {"python": "Python", "numpy": "NumPy", "scipy": "SciPy"}

SKLEARN = "scikit-learn"

# The sections of the maintainer guide on the changes that the report lists
MAINTAINER_GUIDE = "https://alfaro96.github.io/scikit-lr/dev/developers/maintainer.html"
SKLEARN_GUIDE = f"{MAINTAINER_GUIDE}#new-sklearn-release"
VERSIONS_GUIDE = f"{MAINTAINER_GUIDE}#python-numpy-scipy-versions"

# The JSON format of the Index API of PyPI, the one that replaces the list of
# releases of its JSON API, which is deprecated
INDEX_API = "https://pypi.org/simple/{package}/"
INDEX_API_FORMAT = "application/vnd.pypi.simple.v1+json"

JSON_API = "https://pypi.org/pypi/{package}/{version}/json"

PYTHON_RELEASES = "https://www.python.org/api/v2/downloads/release/?is_published=true"

PYTHON_RELEASE_NAME = re.compile(r"^Python (?P<version>3\.\d+)\.0$")

PYTHON_CLASSIFIER = re.compile(
    r"^Programming Language :: Python :: (?P<version>3\.\d+)$"
)

CPYTHON_TAG = re.compile(r"^cp3(?P<minor>\d+)$")


def fetch_json(url, accept="application/json"):
    """Return the JSON document at the URL."""
    request = urllib.request.Request(url, headers={"Accept": accept})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_files(package):
    """Return the files of the package on PyPI that are not yanked, by version."""
    index = fetch_json(INDEX_API.format(package=package), INDEX_API_FORMAT)
    files = defaultdict(list)
    for file in index["files"]:
        if file["yanked"]:
            continue
        filename = file["filename"]
        try:
            if filename.endswith(".whl"):
                version = parse_wheel_filename(filename)[1]
            else:
                version = parse_sdist_filename(filename)[1]
        except (InvalidSdistFilename, InvalidWheelFilename):
            # The Windows installers of the oldest releases
            continue
        files[version].append(file)
    return files


def minor(version):
    """Return the minor release of the version, as a version."""
    return Version(".".join(map(str, Version(str(version)).release[:2])))


def add_years(date, years):
    """Return the date the given number of years later."""
    try:
        return date.replace(year=date.year + years)
    except ValueError:
        # February 29 of a year whose later year is not a leap year
        return date.replace(year=date.year + years, day=28)


def lower_bound(specifier):
    """Return the lower bound of the specifier set, or None if it has none."""
    bounds = [
        Version(spec.version)
        for spec in SpecifierSet(str(specifier))
        if spec.operator in (">=", "~=", "==")
    ]
    return max(bounds, default=None)


def python_releases():
    """Return the date of the first release of each minor release of Python."""
    releases = {}
    for release in fetch_json(PYTHON_RELEASES):
        match = PYTHON_RELEASE_NAME.match(release["name"])
        if match and not release["pre_release"]:
            date = datetime.date.fromisoformat(release["release_date"][:10])
            releases[Version(match["version"])] = date
    return releases


def package_releases(files):
    """Return the date of the first release of each minor release of the package."""
    releases = {}
    for version, version_files in files.items():
        if not version.is_prerelease and version.release[2:] in ((), (0,)):
            date = min(file["upload-time"] for file in version_files)
            releases[minor(version)] = datetime.date.fromisoformat(date[:10])
    return releases


def spec0_minimum(releases, years, date):
    """Return the oldest minor release that SPEC 0 still supports on the date."""
    supported = [
        version
        for version, released in releases.items()
        if add_years(released, years) > date
    ]
    return min(supported, default=None)


def latest_release(files, specifier):
    """Return the latest final release of the package within the specifier set."""
    return max(SpecifierSet(str(specifier)).filter(files), default=None)


def cpython_versions(files):
    """Return the CPython versions that the wheels of a release are built for."""
    versions = set()
    for file in files:
        if file["filename"].endswith(".whl"):
            for tag in parse_wheel_filename(file["filename"])[3]:
                match = CPYTHON_TAG.match(tag.interpreter)
                # The wheels of the free-threaded build, which scikit-lr does not
                # build its wheels for, have their own ABI tag
                if match and tag.abi == tag.interpreter:
                    versions.add(Version(f"3.{match['minor']}"))
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--date",
        type=datetime.date.fromisoformat,
        default=datetime.datetime.now(datetime.UTC).date(),
        help="date of the planned release, in ISO format (default: today, in UTC)",
    )
    date = parser.parse_args().date

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = project["project"]
    dependencies = {
        canonicalize_name(requirement.name): requirement.specifier
        for requirement in map(Requirement, project["dependencies"])
    }
    classifiers = {
        Version(match["version"])
        for match in map(PYTHON_CLASSIFIER.match, project["classifiers"])
        if match
    }

    files = {package: fetch_files(package) for package in (SKLEARN, "numpy", "scipy")}

    sklearn = latest_release(files[SKLEARN], dependencies[SKLEARN])
    sklearn_metadata = fetch_json(JSON_API.format(package=SKLEARN, version=sklearn))
    sklearn_requirements = {
        canonicalize_name(requirement.name): requirement.specifier
        for requirement in map(Requirement, sklearn_metadata["info"]["requires_dist"])
        if requirement.marker is None
    }
    sklearn_requirements["python"] = sklearn_metadata["info"]["requires_python"]

    report = []

    newest_sklearn = latest_release(files[SKLEARN], SpecifierSet(">=0"))
    if newest_sklearn not in dependencies[SKLEARN]:
        report.append(
            f"scikit-learn {newest_sklearn} is released, and scikit-lr supports "
            f"{dependencies[SKLEARN]}: move to it as described in "
            f"[the maintainer guide]({SKLEARN_GUIDE})."
        )

    releases = {"python": python_releases()}
    current = {"python": lower_bound(project["requires-python"])}
    for package in ("numpy", "scipy"):
        releases[package] = package_releases(files[package])
        current[package] = lower_bound(dependencies[package])
    for package, years in SUPPORT_YEARS.items():
        minimum = max(
            spec0_minimum(releases[package], years, date),
            minor(lower_bound(sklearn_requirements[package])),
        )
        if minimum > minor(current[package]):
            report.append(
                f"The minimum supported version of {NAMES[package]} can be raised "
                f"from {current[package]} to {minimum} by {date}, as described in "
                f"[the maintainer guide]({VERSIONS_GUIDE})."
            )

    wheels = cpython_versions(files[SKLEARN][sklearn])
    for package in ("numpy", "scipy"):
        latest = latest_release(files[package], SpecifierSet(">=0"))
        wheels &= cpython_versions(files[package][latest])
    for version in sorted(wheels):
        if version > max(classifiers) and version in releases["python"]:
            report.append(
                f"Python {version} is released, and NumPy, SciPy and "
                f"scikit-learn {sklearn} have wheels for it: add it as described in "
                f"[the maintainer guide]({VERSIONS_GUIDE})."
            )

    for item in report:
        print(f"* {item}")


if __name__ == "__main__":
    main()
