![Docker Cloud Automated build](https://img.shields.io/docker/cloud/automated/alfaro96/scikit-lr.svg)
![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/alfaro96/scikit-lr.svg)

# docker-scikit-lr

Docker image to work on the development of the `scikit-lr` package.

## Build

To build the image from the `Dockerfile`:

```
docker build -t alfaro96/scikit-lr:latest .
```

Alternatively, to pull the image:

```
docker pull alfaro96/scikit-lr:latest
```

## Run

To run the image with the default command (mounting the current directory as workspace):

```
docker run -it -v $(pwd)/:/home/scikit-lr/workspace/ --rm alfaro96/scikit-lr:latest
```

