![Docker Cloud Automated build](https://img.shields.io/docker/cloud/automated/alfaro96/scikit-lr.svg)
![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/alfaro96/scikit-lr.svg)

# docker-scikit-lr

Docker image to execute (and analyze) experiments with the development version of the `scikit-lr` package.

## Build

To build an image from the `Dockerfile`:

```
docker build -t alfaro96/scikit-lr:experiments-latest .
```

Alternatively, to pull the image:

```
docker pull alfaro96/scikit-lr:experiments-latest
```

## Run

To execute the experiments in a main file according with some arguments (mounting the current directory as workspace):

```
docker run -it -v $(pwd)/:/home/scikit-lr/workspace/ \
               -e FILE=main.py \
               -e ARGUMENTS="arguments" \
               --rm alfaro96/scikit-lr:experiments-latest
```

