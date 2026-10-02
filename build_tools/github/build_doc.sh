#!/bin/bash

set -e

python -c "import sklearn; sklearn.show_versions()"

make -C doc html
