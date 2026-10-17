#!/bin/bash

set -e

# A regular install of the checked out commit, so the results are those of its
# version, built without isolation with the packages of the environment of the
# reproduction
python -m pip install --verbose --no-build-isolation --no-deps .
