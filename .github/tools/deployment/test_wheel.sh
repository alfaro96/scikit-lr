#!/bin/bash

set -e

setup_test_venv
install_run $PLAT
teardown_test_venv
