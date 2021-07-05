#!/bin/bash

set -e

echo "Upgrading pip and setuptools."
pip install --upgrade pip setuptools

echo "Installing numpy and scipy."
pip install numpy==$NUMPY_MIN_VERSION scipy==$SCIPY_MIN_VERSION

echo "Installing cython."
pip install cython==$CYTHON_MIN_VERSION

echo "Installing scikit-learn."
pip install scikit-learn==$SCIKIT_LEARN_MIN_VERSION

echo "Installing pytest."
pip install pytest==$PYTEST_MIN_VERSION pytest-xdist

echo "Installing codecov."
pip install --upgrade codecov pytest-cov

pip install --editable .
