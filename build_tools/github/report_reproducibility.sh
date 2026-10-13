#!/bin/bash

set -e

# The cells were computed by other jobs, so Snakemake finds their results and
# only writes the reports and the packages of the environment
cd reproducibility
snakemake --cores "$(nproc)"
