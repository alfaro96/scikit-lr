#!/bin/bash

# This script is meant to be called by the
# "install" step defined in linting.yml

# The behavior of the script is controlled by
# the named step defined in the linting.yml
# in the folder .github/workflows of the project

# Exit immediately if a command
# exits with a non-zero status
set -e

# Update the repositories
apt-get update

# Install and update the dependencies

# Pip
echo "Upgrade pip"
pip install --upgrade pip

# Flake8
echo "Installing flake8"
pip install flake8

# Print the Python and Flake8 versions
python --version
python -c 'import flake8; print("Flake8 {}".format(flake8.__version__))'
