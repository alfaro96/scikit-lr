#!/bin/bash

set -e

echo "Upgrading pip and setuptools"
pip install --upgrade pip setuptools

echo "Installing numpy, scipy and cython"
pip install numpy==$NUMPY_VERSION scipy==$SCIPY_VERSION cython==$CYTHON_VERSION

echo "Installing scikit-learn"
pip install scikit-learn==$SCIKIT_LEARN_VERSION

echo "Installing pytest"
pip install pytest==4.6.4 pytest-cov

echo "Installing codecov"
pip install --upgrade codecov

python --version
python -c "import numpy; print('NumPy {0}'.format(numpy.__version__))"
python -c "import scipy; print('SciPy {0}'.format(scipy.__version__))"
python -c "import cython; print('Cython {0}'.format(cython.__version__))"
python -c "import sklearn; print('Scikit-learn {0}'.format(sklearn.__version__))"

python setup.py develop
