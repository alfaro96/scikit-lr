![Docker Cloud Automated build](https://img.shields.io/docker/cloud/automated/alfaro96/scikit-lr.svg)
![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/alfaro96/scikit-lr.svg)

# docker-scikit-lr

Docker image for the development of the `scikit-lr` Python package.

## Build

To build the image:

```
docker build -t alfaro96/scikit-lr .
```

## Run

To run the image:

```
docker run -it -v $(pwd)/:/home/scikit-lr/workspace/ --rm alfaro96/scikit-lr
```

## Docker Hub

One can easily obtain the image using:

```
docker pull alfaro96/scikit-lr
```
