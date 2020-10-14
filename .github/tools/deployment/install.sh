#!/bin/bash

set -e

echo "Upgrading pip and setuptools."
pip install --upgrade pip setuptools

echo "Installing cibuildwheel"
pip install cibuildwheel
