#!/bin/bash

# This script is meant to be called by the "before_install" step defined in
# ".travis.yml". See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the ".travis.yml" in the top level folder of the project

# In this case, the virtual environment is setup

# If the operating system is Linux, then, install gcc and g++ version 6
# and set them as default compilers
if [ $TRAVIS_OS_NAME == "linux" ]
then
    # Add the repository
    sudo apt-add-repository -y ppa:ubuntu-toolchain-r/test
    # Update the repositories
    sudo apt-get update
    # Install the packages
    sudo apt-get install -y gcc-6 g++-6
    # Set the default compilers		
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 60
    sudo update-alternatives --set gcc /usr/bin/gcc-6
    sudo update-alternatives --set g++ /usr/bin/g++-6
    export CC=gcc
    export CXX=g++
# Otherwise, it is Mac OS X, then, set clang and clang++ as default compilers
else
    export CC=clang
    export CXX=clang++
fi

# Show the compiler versions
$CC --version
$CXX --version

# Conda is a preferable alternative that using apt-get and pip

# Get the Conda version
if [ $TRAVIS_OS_NAME == "osx" ]
then
    CONDA_PLATAFORM=MacOSX
else
    CONDA_PLATAFORM=Linux
fi

# Download Conda and rename it
wget https://repo.continuum.io/miniconda/Miniconda3-latest-${CONDA_PLATAFORM}-x86_64.sh -O miniconda.sh
MINICONDA_PATH="$HOME/miniconda"

# Install and update Conda, setting the "PATH" environment variable accordingly 
bash miniconda.sh -b -p $MINICONDA_PATH
export PATH="$MINICONDA_PATH/bin:$PATH"
hash -r
conda update --yes conda

# Obtain the requirements file
cat requirements/*.txt > requirements/aux_requirements.txt

# If the latest version of the packages must be obtained,
# remove the package version specification
# from the requirements file
if [ $LATEST == 1 ]
then
    cat requirements/aux_requirements.txt | awk -F== '{print $1}' > requirements/requirements.txt
else
    cat requirements/aux_requirements.txt > requirements/requirements.txt
fi

# Install and activate the Conda virtual environment
conda create -n testenv --yes python=$PYTHON_VERSION --file=requirements/requirements.txt
source activate testenv

# Print the installed Python version
python --version

# Install "codecov" with pip,
# since they are not available on Conda
pip install codecov
