#!/bin/bash

# This script is meant to be called by the "Install dependencies" step
# defined in cron.yml. The behaviour of the script is controlled by the named
# step defined in the cron.yml in the folder .github/workflows of the project.

# Exit immediately if a command
# exits with a non-zero status
set -e

# Install gcc-6 and g++-6 since they are
# needed to compile some of the extensions
sudo apt-add-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-6 g++-6

# Set gcc-6 and g++-6 as default
# compilers so that they are used
export CC=/usr/bin/gcc-6
export CXX=/usr/bin/g++-6

# Install and update the dependencies to the latest and nightly versions of the packages
# and thus being able to anticipate future errors before stable versions are released
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
python -c 'import numpy; print("NumPy {}".format(numpy.__version__))'
python -c 'import scipy; print("SciPy {}".format(scipy.__version__))'

# Build scikit-lr in the "install.sh" script
# to collapse the verbose build output in the
# GitHub Actions output when it succeeds
python setup.py develop
