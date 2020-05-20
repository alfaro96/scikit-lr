#!/bin/bash

set -e

pip install virtualenv

source multibuild/common_utils.sh
source multibuild/travis_steps.sh
source extra_functions.sh

# Setup build dependencies
before_install

clean_code $REPO_DIR $BUILD_COMMIT
build_wheel $REPO_DIR $PLAT
