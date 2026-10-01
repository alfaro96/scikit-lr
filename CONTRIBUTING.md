# Contributing to scikit-lr

There are many ways to contribute to scikit-lr. Improving the documentation is no less important than improving the code of the library itself. If you find a typo in the documentation, or have made improvements, do not hesitate to create a GitHub issue or, preferably, submit a GitHub pull request.

There are many other ways to help. In particular, [improving, triaging and investigating issues](https://github.com/alfaro96/scikit-lr/issues) and [reviewing other developers' pull requests](https://github.com/alfaro96/scikit-lr/pulls) are very valuable contributions that decrease the burden on the project maintainers.

Another way to contribute is to report issues you are facing, and give a "thumbs up" on issues that others reported and that are relevant to you. It also helps us if you spread the word: reference the project from your blog and articles, link to it from your website, or simply star it on GitHub to say "I use it".

Note that communications on all channels should respect our [code of conduct](CODE_OF_CONDUCT.md).

## Development environment

scikit-lr contains Cython extensions, so it must be compiled to be used from the source tree. The development environment is declared in `environment.yml` and uses packages from [conda-forge](https://conda-forge.org). With [`micromamba`](https://mamba.readthedocs.io) (or `conda`):

```
git clone https://github.com/alfaro96/scikit-lr.git
cd scikit-lr
micromamba create -f environment.yml
micromamba activate scikit-lr
pip install --no-build-isolation --no-deps -e .
pre-commit install
```

The editable install recompiles the extensions when `sklr` is imported, so there is no need to reinstall after changing a Cython file.

The `meson.build` file of each directory lists the files that are installed, so a new Python module or Cython extension must be added to it. Otherwise it cannot be imported, and the continuous integration reports that it is missing from the wheels.

## Checks

Before submitting a pull request, make sure that the tests and the linters pass:

```
pytest sklr
pre-commit run --all-files
```

The `pre-commit` hooks run `ruff` (lint and format), `pyrefly` (type checking), `cython-lint` and `codespell`, among others.

New code should be covered by the tests. To see the lines that they miss:

```
coverage run -m pytest sklr
coverage report --show-missing
```

The coverage of the Cython extensions is measured every night by the continuous integration, which compiles them with the `linetrace` option of `meson.options`.

## Guidelines

* Follow the [scikit-learn API](https://scikit-learn.org/stable/developers/develop.html): estimators must pass the scikit-learn estimator checks.
* Document the public API with [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings, including runnable examples.
* Add tests for every change. Numerical results should be checked against independent implementations or known properties, with fixed random seeds.
* Write commit messages following [Conventional Commits](https://www.conventionalcommits.org) (`feat:`, `fix:`, `docs:`, `test:`, ...).
