#!/bin/bash

set -e

echo "Upgrading pip and setuptools"
pip install --upgrade pip setuptools

echo "Installing numpy, scipy and cython"
pip install numpy==$NUMPY_VERSION scipy==$SCIPY_VERSION cython==$CYTHON_VERSION

echo "Installing pytest"
pip install pytest==4.6.4 pytest-cov

echo "Installing codecov"
pip install --upgrade codecov

python --version
python -c "import numpy; print('NumPy {0}'.format(numpy.__version__))"
python -c "import scipy; print('SciPy {0}'.format(scipy.__version__))"

# Build scikit-lr in this script to collapse the verbose
# build output in GitHub Actions output when it succeeds
python setup.py develop
