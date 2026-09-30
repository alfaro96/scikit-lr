# scikit-lr

scikit-lr is a Python package for label ranking and partial label ranking, built on top of [scikit-learn](https://scikit-learn.org) and distributed under the MIT license.

In label ranking, each sample is associated with a ranking of a fixed set of labels instead of a single class, and the goal is to learn a model that predicts that ranking for new samples. Partial label ranking extends the problem to rankings with ties. scikit-lr provides estimators and metrics for both problems that follow the scikit-learn API, so they can be used with tools such as pipelines, cross-validation and hyperparameter search.

Website: https://scikit-lr.readthedocs.io

## Installation

### Dependencies

scikit-lr requires:

* Python (>= 3.13)
* NumPy (>= 2.2)
* SciPy (>= 1.15)
* scikit-learn (1.9.x)

scikit-lr is compiled against a specific minor release of scikit-learn, so each release of scikit-lr supports only one minor release of scikit-learn.

### User installation

The easiest way to install scikit-lr is using `pip`:

```
pip install -U scikit-lr
```

## Development

Contributions are welcome. See the [contributing guide](CONTRIBUTING.md) to set up a development environment, and note that all communications should respect our [code of conduct](CODE_OF_CONDUCT.md).

### Important links

* Source code repository: https://github.com/alfaro96/scikit-lr
* Download releases: https://pypi.org/project/scikit-lr
* Issue tracker: https://github.com/alfaro96/scikit-lr/issues

### Source code

You can check the latest sources with the command:

```
git clone https://github.com/alfaro96/scikit-lr.git
```

### Testing

After installation, you can launch the test suite with `pytest`:

```
pytest sklr
```

## Project history

The project was started in 2019 as the Ph.D. thesis of Juan Carlos Alfaro Jiménez, whose advisors are Juan Ángel Aledo Sánchez and José Antonio Gámez Martín.

## License

scikit-lr is distributed under the MIT license. See [COPYING](COPYING).
