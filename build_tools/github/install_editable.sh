#!/bin/bash

set -e

options=()

if [[ "$LINETRACE" == "true" ]]; then
    options+=(--config-settings=setup-args=-Dlinetrace=true)
fi

# An editable install, so the tests import the package from the repository,
# where coverage.py finds the Cython sources. Importing the package rebuilds
# it, so it is built without isolation, with the packages of the environment
python -m pip install --verbose --no-build-isolation --no-deps --editable . "${options[@]}"
