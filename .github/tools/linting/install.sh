#!/bin/bash

# This script is meant to be called by the "Install dependencies" step defined
# in linting.yml. The behaviour of the script is controlled by the named step
# defined in the linting.yml in the folder .github/workflows of the project.

# Exit immediately if a command
# exits with a non-zero status
set -e

# Install and update the dependencies to ensure that the latest
# version of Flake8 is used for testing the linting of the package
echo "Upgrading pip"
pip install --upgrade pip

echo "Installing flake8"
pip install flake8

python --version
python -c 'import flake8; print("Flake8 {}".format(flake8.__version__))'
