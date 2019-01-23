[![Build Status](https://travis-ci.com/alfaro96/PLR.svg?branch=master)](https://travis-ci.com/alfaro96/PLR)
![](https://img.shields.io/pypi/pyversions/plr.svg)
[![PyPI version](https://badge.fury.io/py/plr.svg)](https://badge.fury.io/py/plr)

# plr

`plr` is a Python module for dealing with the Partial Label Ranking problem.

## Prerequisites

`plr` requires:

    * Python (>= 3.6)
    * Numpy (>= 1.15.2)
    * Scipy (>= 1.1.0)

and `Linux` or `Mac OS X` operating systems. `Windows` is not currently supported.

## Installation

The easiest way to install `plr` is using `pip` package:

```
pip install plr
```

## Development

Feel free to contribute to the package, but be sure that the standards are followed.

### Source code

The latest sources can be obtained with the command:

```
git clone https://github.com/alfaro96/plr.git
```

### Setting up a development environment

To setup the development environment, it is strongly recommended to use `docker` tools (from outside the source directory). First, the image must be built.

```
docker build -t alfaro96/plr:development .
```

Or:

```
make docker-build
```

Then, the `docker` container is executed (from outside the source directory) with:

```
docker run -ti -v $(pwd)/:/home/plr/workspace/ --rm alfaro96/plr:development
```

Or:

```
make docker-run
```

In fact, both commands can be executed at once with:

```
make docker
```

Alternatively, one can use `Python` virtual environments (see [https://docs.python.org/3/library/venv.html] for details).

## Testing

After installation, the test suite can be executed, from outside the source directory, with:

```
pytest plr
```

or 

```
make test-code
```

## Authors

    * Alfaro Jiménez, Juan Carlos
    * Aledo Sánchez, Juan Ángel
    * Gámez Martín, José Antonio

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/alfaro96/PLR/blob/master/LICENSE) file for details.
