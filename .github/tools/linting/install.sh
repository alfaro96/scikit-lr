#!/bin/bash

set -e

echo "Upgrading pip."
pip install --upgrade pip

echo "Installing flake8."
pip install flake8
