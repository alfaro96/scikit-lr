#!/bin/bash

conda install -q -y anaconda-client

anaconda -t $ANACONDA_TOKEN upload --force -u alfaro96 dist/artifact/*
echo "Index: https://pypi.anaconda.org/alfaro96/simple"
