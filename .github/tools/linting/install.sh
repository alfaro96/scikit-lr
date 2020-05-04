#!/bin/bash

# Immediately exit with a non-zero status command
set -e

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing flake8"
pip install flake8

python --version
python -c 'import flake8; print("Flake8 {0}".format(flake8.__version__))'
