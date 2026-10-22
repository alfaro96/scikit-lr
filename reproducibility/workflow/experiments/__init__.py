"""Datasets, scenarios, metrics, cross-validation and methods of each paper.

Each paper has a directory named by its citation key, whose ``experiment.py``
declares its protocol in ``PAPER``, collected here in ``PAPERS``. It declares the
tag of its datasets in OpenML and their names there, and its scenarios, each with a
description and the function that transforms the training rankings in it, which
receives them and a random state, or ``None`` to train with them as they are. It
also declares the name of its metric in ``METRICS``, and its cross-validation: the
outer splitter, whose repetitions, if it has several, are run as separate jobs, and,
if some method has a search, the inner splitter that chooses its parameters in each
outer training fold. Finally, it declares the estimator of each method, its fixed
parameters, those that depend on the training data, if any, as a function that
receives the samples and rankings of each training fold and returns them, and the
search that chooses the others with the inner cross-validation, if any, as described
in ``experiments.search``.
"""

import importlib.util
from pathlib import Path

from experiments.datasets import load_dataset
from experiments.metrics import METRICS

__all__ = ["METRICS", "PAPERS", "load_dataset", "make_ranker", "n_repetitions"]


def load_paper(path):
    """Load the protocol of a paper from its ``experiment.py``."""
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PAPER


# Each directory of the reproduction with an experiment.py is a paper, named by
# its citation key
PAPERS = {
    path.parent.name: load_paper(path)
    for path in sorted(Path(__file__).parents[2].glob("*/experiment.py"))
}


def n_repetitions(splitter):
    """Count the repetitions of a splitter, one if it does not repeat its splits."""
    return getattr(splitter, "n_repeats", 1)


def make_ranker(paper, method, n_jobs):
    """Build the ranker of a method of a paper, with its fixed parameters.

    `n_jobs` is passed to the rankers that have that parameter.
    """
    method = PAPERS[paper]["methods"][method]
    ranker = method["estimator"](**method["params"])
    if "n_jobs" in ranker.get_params():
        ranker.set_params(n_jobs=n_jobs)
    return ranker
