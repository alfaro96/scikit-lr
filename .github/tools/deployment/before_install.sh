#!/bin/bash

set -e

# Set the platform variables used in multibuild scripts
if [ $(uname) == "Darwin" ]; then
    echo "::set-env name=TRAVIS_OS_NAME::osx"
    echo "::set-env name=MACOSX_DEPLOYMENT_TARGET::10.9"
else
    echo "::set-env name=TRAVIS_OS_NAME::linux"
fi

# Store the original Python path to be able to create
# a testing environment using the same Python version
echo "::set-env name=PYTHON_EXE::`which python`"
