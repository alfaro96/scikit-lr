"""Check that the wheels contain exactly the files of the package.

The files of a wheel are compared with the ones expected from the files of
``sklr`` tracked by ``git``, so a wheel can neither miss a module that the
``meson.build`` files forgot to install nor carry files that users do not need.
A wheel must have the Python files, except the build helpers in
``sklr/_build_utils``, the ``py.typed`` marker, an extension module for each
Cython source, and the modules that Meson generates. Besides, the tools that
repair the wheels bundle the OpenMP runtime with them, which must be the one of
the platform of the wheel and have its license in the license file of the wheel,
so that a wheel built without OpenMP is also reported. Outside ``sklr`` and that
runtime, only the ``.dist-info`` directory is allowed, so another bundled library
is reported too, since its license would have to be shipped with it.

Run it from the root of the repository with the paths of the wheels.
"""

import subprocess
import sys
import zipfile
from fnmatch import fnmatch
from pathlib import PurePosixPath

PACKAGE = "sklr"

# Directories of the package whose files are only used to build it
BUILD_ONLY_DIRS = [PurePosixPath(PACKAGE, "_build_utils")]

# Files installed as they are, besides the Python modules
DATA_FILES = ["py.typed"]

GENERATED_FILES = [PurePosixPath(PACKAGE, "_build_info.py")]

# Suffixes of the extension modules (on Linux and macOS, and on Windows)
EXTENSION_SUFFIXES = (".so", ".pyd")

# The OpenMP runtime bundled with the wheels of each platform, found by a part of
# the platform tag, and the name under which build_tools/wheels lists its license
OPENMP_RUNTIMES = {
    "linux": ("scikit_lr.libs/libgomp*.so*", "GCC runtime library"),
    "macosx": ("sklr/.dylibs/libomp.dylib", "libomp runtime library"),
    "win": ("scikit_lr.libs/vcomp140*.dll", "Microsoft Visual C++ Runtime Files"),
}


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


def openmp_runtime(wheel):
    """Return the pattern of the OpenMP runtime of `wheel` and its license name."""
    # The platform tag is the last part of the name of a wheel
    platform = PurePosixPath(wheel).stem.rsplit("-", maxsplit=1)[1]
    for key, runtime in OPENMP_RUNTIMES.items():
        if key in platform:
            return runtime
    raise ValueError(f"Unknown platform of the wheel {wheel}, got {platform}.")


def check_wheel(wheel, installed_files, extensions):
    """Return the errors found in the contents of `wheel`."""
    runtime_pattern, runtime_license = openmp_runtime(wheel)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        license_files = [
            name for name in names if fnmatch(name, "*.dist-info/licenses/COPYING")
        ]
        license_text = archive.read(license_files[0]).decode() if license_files else ""
    files = [PurePosixPath(name) for name in names if not name.endswith("/")]
    found_installed_files = set()
    found_extensions = set()
    found_runtime = False
    unexpected = []
    for path in files:
        if path.parts[0].endswith(".dist-info"):
            continue
        stem = extension_stem(path)
        if path in installed_files:
            found_installed_files.add(path)
        elif stem in extensions:
            found_extensions.add(stem)
        elif fnmatch(str(path), runtime_pattern):
            found_runtime = True
        else:
            unexpected.append(path)
    missing = [
        *sorted(installed_files - found_installed_files),
        *(f"{stem}.*" for stem in sorted(extensions - found_extensions)),
    ]
    if not found_runtime:
        missing.append(f"{runtime_pattern} (the OpenMP runtime)")
    errors = [
        *(f"{wheel}: unexpected file {path}" for path in unexpected),
        *(f"{wheel}: missing file {path}" for path in missing),
    ]
    if f"Name: {runtime_license}" not in license_text:
        errors.append(f"{wheel}: missing license of {runtime_license}")
    return errors


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
