[![Continuous integration tests](https://github.com/alfaro96/scikit-lr/workflows/Continuous%20integration%20tests/badge.svg)](https://github.com/alfaro96/scikit-lr/actions?query=workflow%3A%22Continuous+integration+tests%22)
[![Continuous deployment wheels](https://github.com/alfaro96/scikit-lr/workflows/Continuous%20deployment%20wheels/badge.svg)](https://github.com/alfaro96/scikit-lr/actions?query=workflow%3A%22Continuous+deployment+wheels%22)
[![Linting tests](https://github.com/alfaro96/scikit-lr/workflows/Linting%20tests/badge.svg)](https://github.com/alfaro96/scikit-lr/actions?query=workflow%3A%22Linting+tests%22)
[![Daily tests](https://github.com/alfaro96/scikit-lr/workflows/Daily%20tests/badge.svg)](https://github.com/alfaro96/scikit-lr/actions?query=workflow%3A%22Daily+tests%22)
[![Code coverage](https://codecov.io/gh/alfaro96/scikit-lr/branch/master/graph/badge.svg)](https://codecov.io/gh/alfaro96/scikit-lr)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/alfaro96/scikit-lr.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/alfaro96/scikit-lr/context:python)
[![PyPi package](https://badge.fury.io/py/scikit-lr.svg)](https://pypi.org/project/scikit-lr/)
[![Python version](https://img.shields.io/pypi/pyversions/scikit-lr.svg)](https://pypi.org/project/scikit-lr/)

# Scikit-lr

Scikit-lr is a Python module for Label Ranking problems and distributed under
MIT license.

This project was started in 2019 as the Ph.D. Thesis of Juan Carlos Alfaro
Jiménez, whose advisors are Juan Ángel Aledo Sánchez and José Antonio Gámez
Martín.

Website: https://scikit-lr.readthedocs.io/

## Installation

### Dependencies

Scikit-lr requires:

    * Python (>= 3.6)
    * Numpy (>= 1.17.3)
    * SciPy (>= 1.3.2)
    * Scikit-learn (>= 0.23.0)

**Windows is not currently supported.**

### User installation

If you already have a working installation, the
easiest way to install scikit-lr is using ``pip``:

```
pip install -U scikit-lr
```

The documentation includes more detailed [installation instructions](https://scikit-lr.readthedocs.io/en/latest/getting_started/install.html).

## Release history

See the [release history](https://scikit-lr.readthedocs.io/en/latest/whats_new/index.html)
for a history of notable changes to scikit-lr.

## Development

Feel free to contribute to the package, but
be sure that the standards are followed.

### Important links

* Official source code repository: https://github.com/alfaro96/scikit-lr
* Download releases: https://pypi.org/project/scikit-lr/
* Issue tracker: https://github.com/alfaro96/scikit-lr/issues

### Source code

You can check the latest sources with the command:

```
git clone https://github.com/alfaro96/scikit-lr.git
```

### Testing

After installation, you can launch the test suite from outside the
source directory (you will need to have ``pytest`` >= 4.6.4 installed):

```
pytest sklr
```

## Help and support

### Documentation

* HTML documentation (stable release): https://scikit-lr.readthedocs.io/en/stable/index.html
* HTML documentation (development version): https://scikit-lr.readthedocs.io/en/latest/index.html
* FAQ: https://scikit-lr.readthedocs.io/en/stable/getting_started/faq.html

### Communication

* Issue tracker: https://github.com/alfaro96/scikit-lr/issues
* Website: https://scikit-lr.readthedocs.io/
