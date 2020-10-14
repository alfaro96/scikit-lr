#!/bin/bash

set -e

echo "Upgrading pip and setuptools."
pip install --upgrade pip setuptools

echo "Installing numpy, scipy and scikit-learn."
dev_anaconda_url=https://pypi.anaconda.org/scipy-wheels-nightly/simple
pip install --pre --upgrade --timeout=60 --extra-index $dev_anaconda_url numpy scipy scikit-learn

echo "Installing cython."
pip install --pre --upgrade --timeout=60 cython

echo "Installing pytest."
pip install pytest==4.6.4 pytest-cov

echo "Installing codecov."
pip install --upgrade codecov

python setup.py develop
