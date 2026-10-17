#!/bin/bash

set -e

# Run the tests of the installed package from outside the repository,
# so the sources of the checkout are not imported by mistake
test_dir="$RUNNER_TEMP/tests"
mkdir -p "$test_dir"

# pytest only reads its configuration from the directory where it runs and
# its parents, so without a copy of pyproject.toml the tests would run
# without the options of the project. The pages of the documentation are
# copied as well, to run their examples against the installed package
cp -r pyproject.toml doc "$test_dir"
cd "$test_dir"

python -c "import sklr; sklr.show_versions()"

python -m pytest --pyargs sklr doc
