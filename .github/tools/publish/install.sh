#!/bin/bash

set -e

echo "Upgrading pip."
pip install --upgrade pip

echo "Installing wheelhouse-uploader."
pip install wheelhouse-uploader
