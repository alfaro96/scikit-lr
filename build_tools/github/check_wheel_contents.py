"""Check that the wheels contain exactly the files of the package.

The files of a wheel are compared with the ones expected from the files of
``sklr`` tracked by ``git``, so a wheel can neither miss a module that the
``meson.build`` files forgot to install nor carry files that users do not need.
A wheel must have the Python files, except the build helpers in
``sklr/_build_utils``, the ``py.typed`` marker, an extension module for each
Cython source, and the modules that Meson generates. Outside ``sklr``, only its
``.dist-info`` directory is allowed, so a library bundled by the tools that
repair the wheels is also reported, since its license would have to be shipped
with them.

Run it from the root of the repository with the paths of the wheels.
"""

import subprocess
import sys
import zipfile
from pathlib import PurePosixPath

PACKAGE = "sklr"

# Directories of the package whose files are only used to build it
BUILD_ONLY_DIRS = [PurePosixPath(PACKAGE, "_build_utils")]

# Files installed as they are, besides the Python modules
DATA_FILES = ["py.typed"]

GENERATED_FILES = [PurePosixPath(PACKAGE, "_build_info.py")]

# Suffixes of the extension modules (on Linux and macOS, and on Windows)
EXTENSION_SUFFIXES = (".so", ".pyd")


def tracked_files():
    """Return the files of the package tracked by ``git``."""
    output = subprocess.run(
        ["git", "ls-files", "-z", PACKAGE],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return [PurePosixPath(path) for path in output.split("\0") if path]


def expected_files(tracked):
    """Return the files and the extension modules expected in a wheel.

    The extension modules are returned without their suffix, which depends on
    the Python version and the platform.
    """
    sources = [
        path
        for path in tracked
        if not any(path.is_relative_to(directory) for directory in BUILD_ONLY_DIRS)
    ]
    installed_files = {
        path for path in sources if path.suffix == ".py" or path.name in DATA_FILES
    }
    installed_files.update(GENERATED_FILES)
    extensions = {path.with_suffix("") for path in sources if path.suffix == ".pyx"}
    return installed_files, extensions


def extension_stem(path):
    """Return the path of an extension module without its suffix, or ``None``."""
    if path.suffix not in EXTENSION_SUFFIXES:
        return None
    # The suffix has several dots
    return path.with_name(path.name.split(".", maxsplit=1)[0])


def check_wheel(wheel, installed_files, extensions):
    """Return the errors found in the contents of `wheel`."""
    with zipfile.ZipFile(wheel) as archive:
        files = [
            PurePosixPath(name) for name in archive.namelist() if not name.endswith("/")
        ]
    found_installed_files = set()
    found_extensions = set()
    unexpected = []
    for path in files:
        if path.parts[0].endswith(".dist-info"):
            continue
        stem = extension_stem(path)
        if path in installed_files:
            found_installed_files.add(path)
        elif stem in extensions:
            found_extensions.add(stem)
        else:
            unexpected.append(path)
    missing = [
        *sorted(installed_files - found_installed_files),
        *(f"{stem}.*" for stem in sorted(extensions - found_extensions)),
    ]
    return [
        *(f"{wheel}: unexpected file {path}" for path in unexpected),
        *(f"{wheel}: missing file {path}" for path in missing),
    ]


def main(wheels):
    if not wheels:
        print("Pass the paths of the wheels to check", file=sys.stderr)
        return 1
    installed_files, extensions = expected_files(tracked_files())
    errors = []
    for wheel in wheels:
        errors.extend(check_wheel(wheel, installed_files, extensions))
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print(f"Checked {len(wheels)} wheel(s), all with the expected files")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
