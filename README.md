[![Integration](https://github.com/alfaro96/scikit-lr/workflows/Continuous%20integration%20tests/badge.svg)](https://github.com/alfaro96/scikit-lr/actions?query=workflow%3A%22Continuous+integration+tests%22)
[![Linting](https://github.com/alfaro96/scikit-lr/workflows/Linting%20tests/badge.svg)](https://github.com/alfaro96/scikit-lr/actions?query=workflow%3A%22Linting+tests%22)
[![CRON](https://github.com/alfaro96/scikit-lr/workflows/Daily%20wheels%20tests/badge.svg)](https://github.com/alfaro96/scikit-lr/actions?query=workflow%3A%22Daily+wheels+tests%22)
[![Codecov](https://codecov.io/gh/alfaro96/scikit-lr/branch/master/graph/badge.svg)](https://codecov.io/gh/alfaro96/scikit-lr)
[![PyPI](https://badge.fury.io/py/scikit-lr.svg)](https://pypi.org/project/scikit-lr/)
[![Python](https://img.shields.io/pypi/pyversions/scikit-lr.svg)](https://pypi.org/project/scikit-lr/)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

# scikit-lr

`scikit-lr` is a Python module integrating Machine Learning algorithms for Label Ranking problems and distributed under MIT license.

## Installation

### Dependencies

`scikit-lr` requires:

    * Python>=3.6
    * Numpy>=1.15.2
    * SciPy>=1.1.0

`Linux` or `Mac OS X` operating systems. `Windows` is not currently supported.

### User installation

The easiest way to install `scikit-lr` is using `pip` package:

```
pip install -U scikit-lr
```

## Development

Feel free to contribute to the package, but be sure that the standards are followed.

### Source code

The latest sources can be obtained with the command:

```
git clone https://github.com/alfaro96/scikit-lr.git
```

### Setting up a development environment

To setup the development environment, it is strongly recommended to use `docker` tools (see https://github.com/alfaro96/docker-scikit-lr for details).

Alternatively, one can use `Python` virtual environments (see https://docs.python.org/3/library/venv.html for details).

### Testing

After installation the test suite can be executed from outside the source directory, with (you will need to have `pytest>=4.6.4` installed):

```
pytest sklr
```

## Authors

    * Alfaro Jiménez, Juan Carlos
    * Aledo Sánchez, Juan Ángel
    * Gámez Martín, José Antonio

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/alfaro96/scikit-lr/blob/master/LICENSE) file for details.
