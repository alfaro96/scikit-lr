#!/bin/bash

set -e

python -m pip install pytest packaging

options=()

if [[ "$DEPENDENCIES" == "minimum" ]]; then
    requirements="$RUNNER_TEMP/minimum-requirements.txt"
    python build_tools/github/minimum_requirements.py > "$requirements"
    cat "$requirements"
    options+=(--requirement "$requirements")
elif [[ "$DEPENDENCIES" == "conda-forge" ]]; then
    # Keep the dependencies of the environment, which pip would
    # otherwise replace with the ones of PyPI
    options+=(--no-deps)
fi

if [[ "$RUNNER_OS" == "Windows" ]]; then
    # Use MSVC, the compiler of CPython on Windows, instead of the
    # MinGW compiler that the runner also has in the PATH
    options+=(--config-settings=setup-args=--vsenv)
fi

# A regular (not editable) install in an isolated build environment,
# to test the package as users get it
python -m pip install --verbose . "${options[@]}"

if [[ "$DEPENDENCIES" == "conda-forge" ]]; then
    # Fail if the environment does not satisfy the requirements of scikit-lr,
    # which pip did not check without installing its dependencies
    python -m pip check
fi
