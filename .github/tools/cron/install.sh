#!/bin/bash

# Immediately exit with a non-zero status command
set -e

# Install GCC and G++ to compile the Cython extensions
sudo apt-add-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-6 g++-6

export CC=/usr/bin/gcc-6
export CXX=/usr/bin/g++-6

# Use the latest and nightly versions of the packages
# to anticipate future errors before stable releases
echo "Upgrading pip and setuptools"
pip install --upgrade pip setuptools

echo "Installing numpy, scipy and cython"
dev_url=https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com
pip install --pre --upgrade --timeout=60 -f $dev_url numpy scipy cython

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
