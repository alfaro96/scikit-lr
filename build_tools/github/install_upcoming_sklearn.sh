#!/bin/bash

set -e

python -m pip install pytest packaging

# Building without isolation needs the build requirements installed, and
# also ninja, which meson-python only adds to an isolated build environment
requirements="$RUNNER_TEMP/build-requirements.txt"
python build_tools/github/build_requirements.py > "$requirements"
cat "$requirements"
python -m pip install --requirement "$requirements" ninja

# The latest stable scikit-learn brings its dependencies, and then only
# scikit-learn is upgraded to the upcoming release, so a failure points to
# it and not to a pre-release of NumPy or SciPy. Only wheels are installed,
# which is how users get scikit-learn
python -m pip install scikit-learn
options=(--pre --upgrade --no-deps --only-binary :all:)
if [[ "$SKLEARN" == "nightly" ]]; then
    options+=(--index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple)
fi
python -m pip install "${options[@]}" scikit-learn

# The nightly index also has the stable releases, which pip keeps installed
# when it finds no nightly wheel for this platform, so the job would test
# the stable scikit-learn without failing
version=$(python -c "import sklearn; print(sklearn.__version__)")
if [[ "$SKLEARN" == "nightly" && "$version" != *.dev* ]]; then
    echo "Expected a nightly release of scikit-learn, got $version" >&2
    exit 1
fi

# Fail if the upcoming scikit-learn needs newer dependencies than the ones
# installed. It runs before installing scikit-lr, whose requirements may not
# allow that scikit-learn
python -m pip check

# Without isolation, so the extension modules are compiled against the
# upcoming scikit-learn, and without dependencies, which would otherwise
# replace it with the minor release supported by scikit-lr
python -m pip install --verbose --no-build-isolation --no-deps .
