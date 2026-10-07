#!/bin/bash

# Append to the license of scikit-lr the licenses of the libraries that the
# repair of the wheels bundles with them, so that the wheels distribute them

set -e

PROJECT_DIR=$1
BUNDLED_LICENSES=$2
LICENSE_FILE="$PROJECT_DIR/COPYING"

# cibuildwheel runs this script before building each wheel, in the same
# directory on macOS and Windows, so the licenses are appended only once
if ! grep -q "also bundles the following software" "$LICENSE_FILE"; then
    printf "\n----\n\n" >>"$LICENSE_FILE"
    cat "$PROJECT_DIR/$BUNDLED_LICENSES" >>"$LICENSE_FILE"
fi
