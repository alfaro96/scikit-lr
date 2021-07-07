#!/bin/bash

set -e

echo "Downloading scikit-lr $SCIKIT_LR_VERSION wheels and source distribution."
python -m wheelhouse_uploader fetch --version $SKLEARN_VERSION \
                                    --local-folder dist \
                                    scikit-learn \
                                    https://pypi.anaconda.org/alfaro/simple/scikit-lr
