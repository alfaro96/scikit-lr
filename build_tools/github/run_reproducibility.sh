#!/bin/bash

set -e

# The cells of the reproduction are split into N_BATCHES batches, and this job
# runs the cells of batch BATCH, each one with a single thread, as many at a
# time as the runner has cores
cd reproducibility
snakemake cells --cores "$(nproc)" \
    --config "batch=$BATCH/$N_BATCHES" "data_home=$DATA_HOME"
