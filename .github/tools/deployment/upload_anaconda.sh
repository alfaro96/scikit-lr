#!/bin/bash

conda install -q -y anaconda-client

ls wheelhouse/*.whl

anaconda -t $CONDA_TOKEN upload --force -u alfaro96 dist/artifact/*
echo "Index: https://pypi.anaconda.org/alfaro96/simple"
