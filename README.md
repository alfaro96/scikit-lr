[![Travis](https://travis-ci.com/alfaro96/plr.svg?branch=master)](https://travis-ci.com/alfaro96/plr)
[![Codecov](https://codecov.io/gh/alfaro96/plr/branch/master/graph/badge.svg)](https://codecov.io/gh/alfaro96/plr)
[![PyPI](https://badge.fury.io/py/plr.svg)](https://badge.fury.io/py/plr)
![Python](https://img.shields.io/pypi/pyversions/plr.svg)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

# plr

`plr` is a Python module integrating Machine Learning algorithms for Label Ranking problems and distributed under MIT license.

## Installation

### Dependencies

`plr` requires:

    * Python>=3.6
    * Numpy>=1.15.2
    * SciPy>=1.1.0

`Linux` or `Mac OS X` operating systems. `Windows` is not currently supported.

### User installation

The easiest way to install `plr` is using `pip` package:

```
pip install -U plr
```

## Development

Feel free to contribute to the package, but be sure that the standards are followed.

### Source code

The latest sources can be obtained with the command:

```
git clone https://github.com/alfaro96/plr.git
```

### Setting up a development environment

To setup the development environment, it is strongly recommended to use `docker` tools (see [https://github.com/alfaro96/docker-plr]) 

Alternatively, one can use `Python` virtual environments (see [https://docs.python.org/3/library/venv.html] for details).

### Testing

After installation the test suite can be executed from outside the source directory, with (you will need to have `pytest>=4.0.1` installed):

```
pytest plr
```

## Authors

    * Alfaro Jiménez, Juan Carlos
    * Aledo Sánchez, Juan Ángel
    * Gámez Martín, José Antonio

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/alfaro96/PLR/blob/master/LICENSE) file for details.
