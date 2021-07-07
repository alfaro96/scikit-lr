#!/bin/bash

set -e

echo "Downloading scikit-lr $SCIKIT_LR_VERSION wheels and source distribution."
python -m wheelhouse_uploader fetch --version $SCIKIT_LR_VERSION \
                                    --local-folder dist \
                                    scikit-lr \
                                    https://pypi.anaconda.org/alfaro96/simple/scikit-lr
