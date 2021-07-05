#!/bin/bash

set -e

echo "Upgrading pip and setuptools."
pip install --upgrade pip setuptools

echo "Installing numpy, scipy and scikit-learn."
dev_anaconda_url=https://pypi.anaconda.org/scipy-wheels-nightly/simple
pip install --pre --extra-index $dev_anaconda_url numpy scipy scikit-learn

echo "Installing cython."
pip install --pre cython

echo "Installing pytest."
pip install pytest pytest-xdist

echo "Installing codecov."
pip install codecov pytest-cov

pip install --editable .
