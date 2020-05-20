#!/bin/bash

set -e

source multibuild/common_utils.sh
source multibuild/travis_steps.sh
source extra_functions.sh

setup_test_venv
install_run $PLAT
teardown_test_venv
