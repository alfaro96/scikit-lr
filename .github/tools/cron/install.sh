#!/bin/bash

set -e

echo "Upgrading pip and setuptools."
pip install --upgrade pip setuptools

echo "Installing numpy, scipy and scikit-learn."
dev_anaconda_url=https://pypi.anaconda.org/scipy-wheels-nightly/simple
pip install --pre --upgrade --timeout=60 --extra-index $dev_anaconda_url numpy scipy scikit-learn

# Cython nightly builds should be still fetched from the Rackspace container
echo "Installing cython."
dev_rackspace_url=https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com
pip install --pre --upgrade --timeout=60 -f $dev_rackspace_url cython

echo "Installing pytest."
pip install pytest==4.6.4 pytest-cov

echo "Installing codecov."
pip install --upgrade codecov

python setup.py develop
