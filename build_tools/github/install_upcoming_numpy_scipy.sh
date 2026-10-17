#!/bin/bash

set -e

python -m pip install pytest packaging

# Building without isolation needs the build requirements installed, and
# also ninja, which meson-python only adds to an isolated build environment.
# They include the minor release of scikit-learn supported by scikit-lr
requirements="$RUNNER_TEMP/build-requirements.txt"
python build_tools/github/build_requirements.py > "$requirements"
cat "$requirements"
python -m pip install --only-binary :all: --requirement "$requirements" ninja

# Then only NumPy and SciPy are upgraded to their nightly wheels, so a failure
# points to them and not to a pre-release of another package. Only wheels are
# installed, which is how users get them
python -m pip install --pre --upgrade --no-deps --only-binary :all: \
    --index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    numpy scipy

# The nightly index also has the stable releases, which pip keeps installed
# when it finds no nightly wheel for this platform, so the job would test
# the stable releases without failing
for package in numpy scipy; do
    version=$(python -c "import importlib.metadata; print(importlib.metadata.version('$package'))")
    if [[ "$version" != *.dev* ]]; then
        echo "Expected a nightly release of $package, got $version" >&2
        exit 1
    fi
done

# Fail if the nightly NumPy and SciPy need newer dependencies than the ones
# installed, or if the installed packages do not allow them
python -m pip check

# Without isolation, so the extension modules are compiled against the
# nightly SciPy, whose Cython API they cimport, and without dependencies,
# which would otherwise replace NumPy and SciPy with their stable releases
python -m pip install --verbose --no-build-isolation --no-deps .
