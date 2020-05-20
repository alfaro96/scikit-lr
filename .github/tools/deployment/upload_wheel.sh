#!/bin/bash

# Add Conda directory to PATH environment variable
echo ::add-path::$CONDA/bin
sudo chown -R $USER $CONDA

conda install -q -y anaconda-client

# Force a replacement if the remote file already exists
ls wheelhouse/*.whl
anaconda -t $CONDA_TOKEN upload --force -u alfaro96 wheelhouse/*.whl
