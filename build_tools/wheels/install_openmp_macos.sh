#!/bin/bash

# Install the OpenMP runtime of LLVM, which the compiler of macOS lacks, from
# conda-forge. Its version is the oldest one built for each architecture, so
# that the runtime bundled with the wheels supports the oldest macOS version
# that they support

set -e

PREFIX=$1

if [[ $(uname -m) == "arm64" ]]; then
    PACKAGE="osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"
    SHA256="29763e493e46801c416f2e75f20e9183719bde3b727d8983b1917a480b4076fe"
else
    PACKAGE="osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
    SHA256="1ebfeee2af90bcedf6b39c7290244e2aec0bda01ecf81b0d8d78ed405480c21c"
fi

curl -fsSL -o llvm-openmp.tar.bz2 \
    "https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/$PACKAGE"
echo "$SHA256  llvm-openmp.tar.bz2" | shasum -a 256 --check
mkdir -p "$PREFIX"
tar -xjf llvm-openmp.tar.bz2 -C "$PREFIX"
rm llvm-openmp.tar.bz2
