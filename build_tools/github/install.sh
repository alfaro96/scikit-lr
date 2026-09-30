#!/bin/bash

set -e

# pytest also installs packaging, used to read the minimum versions
python -m pip install pytest

options=()

if [[ "$DEPENDENCIES" == "minimum" ]]; then
    requirements="$RUNNER_TEMP/minimum-requirements.txt"
    python build_tools/github/minimum_requirements.py > "$requirements"
    cat "$requirements"
    options+=(--requirement "$requirements")
fi

if [[ "$RUNNER_OS" == "Windows" ]]; then
    # Use MSVC, the compiler of CPython on Windows, instead of the
    # MinGW compiler that the runner also has in the PATH
    options+=(--config-settings=setup-args=--vsenv)
fi

# A regular (not editable) install in an isolated build environment,
# to test the package as users get it
python -m pip install --verbose . "${options[@]}"
