#!/bin/bash

set -e

python -c "import sklearn; sklearn.show_versions()"

# Run the tests of the installed package from outside the repository,
# so the sources of the checkout are not imported by mistake
test_dir="$RUNNER_TEMP/tests"
mkdir -p "$test_dir"

# pytest only reads its configuration from the directory where it runs and
# its parents, so without a copy of pyproject.toml the tests would run
# without the options of the project
cp pyproject.toml "$test_dir"
cd "$test_dir"

python -m pytest --pyargs sklr
