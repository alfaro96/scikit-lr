![Docker Cloud Automated Build](https://img.shields.io/docker/cloud/automated/alfaro96/scikit-lr.svg)
![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/alfaro96/scikit-lr.svg)
![Docker Pulls](https://img.shields.io/docker/pulls/alfaro96/scikit-lr.svg)
![Docker Image Size](https://img.shields.io/docker/image-size/alfaro96/scikit-lr/latest.svg)

# Using `scikit-lr` via `Docker`

This directory contains a `Dockerfile` to make it easy
to get up and running with `scikit-lr` via `Docker`.

## Pulling the image

To pull the image from [Docker Hub](https://hub.docker.com/r/alfaro96/scikit-lr):

```
docker pull alfaro96/scikit-lr:latest
```

## Running the container

To start a `bash`:

```
docker run -it \
           -v $(PWD)/:/home/scikit-lr/workspace \
           --rm alfaro96/scikit-lr:latest \
           /bin/bash
```

To start a `notebook`:

```
docker run -it \
           -p 8888:8888 \
           -v $(PWD)/:/home/scikit-lr/workspace \
           --rm alfaro96/scikit-lr:latest \
           jupyter lab --ip=0.0.0.0
```
