#!/bin/bash

set -e

source multibuild/common_utils.sh
source multibuild/travis_steps.sh
source extra_functions.sh

clean_code $REPO_DIR $BUILD_COMMIT
build_wheel $REPO_DIR $PLAT
