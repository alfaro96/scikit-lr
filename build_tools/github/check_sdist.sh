#!/bin/bash

set -e

python -m pip install check-sdist

# Compare the files of the source distribution with the ones tracked by git,
# adding untracked files to the checkout to check that they are left out
check-sdist --inject-junk
