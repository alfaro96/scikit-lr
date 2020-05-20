#!/bin/bash

conda install -q -y anaconda-client

# Force a replacement if the remote file already exists
ls wheelhouse/*.whl
anaconda -t $CONDA_TOKEN upload --force -u alfaro96 wheelhouse/*.whl
