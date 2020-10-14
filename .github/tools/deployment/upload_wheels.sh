#!/bin/bash

conda install -q -y anaconda-client

ls wheelhouse/*.whl

# Force a replacement if the remote file already exists
anaconda -t $CONDA_TOKEN upload --force -u alfaro96 wheelhouse/*.whl
