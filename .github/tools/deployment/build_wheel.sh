#!/bin/bash

set -e

clean_code $REPO_DIR $BUILD_COMMIT
build_wheel $REPO_DIR $PLAT
