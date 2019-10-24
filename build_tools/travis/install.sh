#!/bin/bash

# This script is meant to be called by the "install" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.

# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# Travis clone alfaro96/plr into a local repository.

# We use a cached directory with plr repositories (one for each
# matrix entry) from which we pull from local Travis repository.
# This allows us to keep build artefact for gcc + Cython, and gain time.

# Exit immediately if a command exits with a non-zero status
set -e

# List files from the directories that have been cached
echo "List files from cached directories"
echo "pip: "
ls $HOME/.cache/pip

# Install gcc-6 and g++-6, since it is needed by some extension modules of Cython

# Add the repository
sudo apt-add-repository -y ppa:ubuntu-toolchain-r/test

# Update the repositories
sudo apt-get update

# Install the packages
sudo apt-get install -y gcc-6 g++-6

# Set the default compilers
export CC=/usr/bin/gcc-6
export CXX=/usr/bin/g++-6

# Deactivate the Travis-provided virtual environment and
# setup a Conda-based environment instead. If Travis has
# language=generic, deactivate does not exist. `|| :` will pass.
deactivate || :

# Install Conda
fname=Miniconda3-latest-Linux-x86_64.sh
wget https://repo.continuum.io/miniconda/$fname -O miniconda.sh
MINICONDA_PATH=$HOME/miniconda
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH

# Add Conda to the "PATH" environment variable
export PATH=$MINICONDA_PATH/bin:$PATH

# Update Conda
conda update --yes conda

# Create the virtual environment
conda create -n testenv --yes python=3.7
source activate testenv

# Install and update the dependencies

# Setuptools
echo "Installing setuptools"
pip install --upgrade pip setuptools

# NumPy and Scipy
echo "Installing NumPy and SciPy"
pip install --upgrade numpy scipy

# Cython		
echo "Installing Cython"		
pip install --upgrade cython

# Pytest
echo "Installing pytest"
pip install pytest==4.6.4 pytest-cov

# Codecov
echo "Installing codecov"
pip install --upgrade codecov

# Print the Python, NumPy and Scipy versions
python --version
python -c 'import numpy; print("NumPy {}".format(numpy.__version__))'
python -c 'import scipy; print("SciPy {}".format(scipy.__version__))'

# Build plr in the install.sh script to collapse the verbose
# build output in the Travis output when it succeeds.
python setup.py develop
