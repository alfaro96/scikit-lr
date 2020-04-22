![Docker Cloud Automated Build](https://img.shields.io/docker/cloud/automated/alfaro96/scikit-lr.svg)
![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/alfaro96/scikit-lr.svg)
![Docker Pulls](https://img.shields.io/docker/pulls/alfaro96/scikit-lr.svg)
![Docker Image Size](https://img.shields.io/docker/image-size/alfaro96/scikit-lr/latest.svg)

# docker-scikit-lr

Docker image to work on the development of the `scikit-lr` package.

## Build

To build the image from the `Dockerfile`:

```
make build
```

Alternatively, to pull the image from `Docker Hub`:

```
make pull
```

## Run

To run the image with the default command (mounting the current directory as workspace):

```
make run
```

