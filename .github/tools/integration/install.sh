#!/bin/bash

# Immediately exit with a non-zero status command
set -e

# Install GCC and G++ in Linux to compile the Cython extensions
if [ $(uname | tr "[:upper:]" "[:lower:]") == "linux" ]; then
    sudo apt-add-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install -y gcc-6 g++-6
    export CC=/usr/bin/gcc-6
    export CXX=/usr/bin/g++-6
fi

# Use the specified versions of the packages to ensure
# that all the operating systems use the same version
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
