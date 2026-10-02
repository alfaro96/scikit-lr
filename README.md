# scikit-lr

[![Unit tests][unit-tests-badge]][unit-tests] [![Code quality checks][code-quality-badge]][code-quality] [![Wheels][wheels-badge]][wheels] [![Upcoming scikit-learn][upcoming-sklearn-badge]][upcoming-sklearn] [![Coverage][coverage-badge]][coverage] [![Ruff][ruff-badge]][ruff] [![PyPI][pypi-badge]][pypi] [![Python versions][python-badge]][pypi]

[unit-tests-badge]: https://github.com/alfaro96/scikit-lr/actions/workflows/unit-tests.yml/badge.svg?branch=master
[unit-tests]: https://github.com/alfaro96/scikit-lr/actions/workflows/unit-tests.yml?query=branch%3Amaster
[code-quality-badge]: https://github.com/alfaro96/scikit-lr/actions/workflows/code-quality.yml/badge.svg?branch=master
[code-quality]: https://github.com/alfaro96/scikit-lr/actions/workflows/code-quality.yml?query=branch%3Amaster
[wheels-badge]: https://github.com/alfaro96/scikit-lr/actions/workflows/wheels.yml/badge.svg?branch=master
[wheels]: https://github.com/alfaro96/scikit-lr/actions/workflows/wheels.yml?query=branch%3Amaster
[upcoming-sklearn-badge]: https://github.com/alfaro96/scikit-lr/actions/workflows/upcoming-sklearn.yml/badge.svg?event=schedule
[upcoming-sklearn]: https://github.com/alfaro96/scikit-lr/actions/workflows/upcoming-sklearn.yml?query=event%3Aschedule
[coverage-badge]: https://codecov.io/gh/alfaro96/scikit-lr/branch/master/graph/badge.svg
[coverage]: https://codecov.io/gh/alfaro96/scikit-lr
[ruff-badge]: https://img.shields.io/badge/code%20style-ruff-000000.svg
[ruff]: https://github.com/astral-sh/ruff
[pypi-badge]: https://img.shields.io/pypi/v/scikit-lr
[python-badge]: https://img.shields.io/pypi/pyversions/scikit-lr
[pypi]: https://pypi.org/project/scikit-lr

scikit-lr is a Python package for label ranking and partial label ranking, built on top of [scikit-learn](https://scikit-learn.org) and distributed under the MIT license.

In label ranking, each sample is associated with a ranking of a fixed set of labels instead of a single class, and the goal is to learn a model that predicts that ranking for new samples. Partial label ranking extends the problem to rankings with ties. scikit-lr provides estimators and metrics for both problems that follow the scikit-learn API, so they can be used with tools such as pipelines, cross-validation and hyperparameter search.

Website: https://alfaro96.github.io/scikit-lr

## Installation

### Dependencies

scikit-lr requires:

* Python (>= 3.13)
* NumPy (>= 2.2)
* SciPy (>= 1.15)
* scikit-learn (1.9.x)

### User installation

The easiest way to install scikit-lr is using `pip`:

```
pip install -U scikit-lr
```

The documentation includes more detailed [installation instructions](https://alfaro96.github.io/scikit-lr/dev/install.html).

## Documentation

* [User guide](https://alfaro96.github.io/scikit-lr/dev/user_guide.html)
* [API reference](https://alfaro96.github.io/scikit-lr/dev/api/index.html)
* [Examples](https://alfaro96.github.io/scikit-lr/dev/auto_examples/index.html)

## Changelog

See the [release history](https://alfaro96.github.io/scikit-lr/dev/whats_new.html) for a history of notable changes to scikit-lr.

## Development

Contributions are welcome. The [developer's guide](https://alfaro96.github.io/scikit-lr/dev/developers/index.html) has detailed information about contributing code, documentation and tests, and about the Cython extensions. All communications should respect our [code of conduct](CODE_OF_CONDUCT.md).

### Important links

* Source code repository: https://github.com/alfaro96/scikit-lr
* Download releases: https://pypi.org/project/scikit-lr
* Issue tracker: https://github.com/alfaro96/scikit-lr/issues

### Source code

You can check the latest sources with the command:

```
git clone https://github.com/alfaro96/scikit-lr.git
```

### Contributing

To learn more about making a contribution to scikit-lr, see our [contributing guide](https://alfaro96.github.io/scikit-lr/dev/developers/contributing.html).

### Testing

After installation, you can launch the test suite with `pytest`:

```
pytest --pyargs sklr
```

See [testing and improving test coverage](https://alfaro96.github.io/scikit-lr/dev/developers/contributing.html#testing-and-improving-test-coverage) for more information.

## Project history

The project was started in 2019 as the Ph.D. thesis of Juan Carlos Alfaro Jiménez, whose advisors are Juan Ángel Aledo Sánchez and José Antonio Gámez Martín.

## License

scikit-lr is distributed under the MIT license. See [`COPYING`](COPYING).
