"""Run one repetition of the cross-validation of a method on a dataset and scenario.

In each fold of the outer cross-validation of the paper, the training rankings are
transformed as the scenario says, the parameters of the method that depend on the
training data are set from them, the others are chosen by its search with the
inner cross-validation of the paper, if it has one, and the method is scored with
the metric of the paper on the test rankings. The results are written as JSON to
the output path.
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from experiments import METRICS, PAPERS, load_dataset, make_ranker, n_repetitions
from experiments.search import describe as describe_search
from machine import describe
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

import sklr


def timed(function, *args):
    """Call a function and return its result, wall time and CPU time in seconds."""
    wall, cpu = time.perf_counter(), time.process_time()
    result = function(*args)
    return result, time.perf_counter() - wall, time.process_time() - cpu


def choose_params(ranker, search, scoring, cv, X, y):
    """Choose the parameters with the search of the method, if it has one.

    Returns the chosen parameters and the mean inner score of each candidate, or no
    parameters and scores without a search.
    """
    if search is None:
        return {}, []
    return search(ranker, scoring, cv, X, y)


def run_fold(paper, ranker, data_params, search, transform, X, y, train, test, seed):
    """Choose the parameters on a training fold and score the test fold.

    The parameters that depend on the training data are set from the training fold
    before the inner cross-validation, which keeps them in its own folds.
    """
    metric = METRICS[paper["metric"]]
    y_train = y[train]
    if transform is not None:
        y_train = transform(y_train, np.random.RandomState(seed))
    fold_params = {} if data_params is None else data_params(X[train], y_train)
    ranker = clone(ranker).set_params(**fold_params)
    # The convergence warnings are counted, so that the report shows how often the
    # estimation of the models stops at max_iter
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        (best_params, inner_scores), search_wall, search_cpu = timed(
            choose_params,
            ranker,
            search,
            metric["inner_scorer"],
            paper["cross_validation"]["inner"],
            X[train],
            y_train,
        )
        # Refitted here, instead of by the search, to measure the time of fit
        best = clone(ranker).set_params(**best_params)
        _, fit_wall, fit_cpu = timed(best.fit, X[train], y_train)
        y_pred, predict_wall, predict_cpu = timed(best.predict, X[test])
    return {
        "score": metric["score"](y[test], y_pred),
        "data_params": fold_params,
        "best_params": best_params,
        "inner_scores": inner_scores,
        "unranked_fraction": float(np.isnan(y_train).mean()),
        "convergence_warnings": sum(
            issubclass(w.category, ConvergenceWarning) for w in caught
        ),
        "time": {
            "search_wall": search_wall,
            "search_cpu": search_cpu,
            "fit_wall": fit_wall,
            "fit_cpu": fit_cpu,
            "predict_wall": predict_wall,
            "predict_cpu": predict_cpu,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--repetition", required=True, type=int)
    parser.add_argument("--n-jobs", type=int, default=1)
    # An empty path means the default directory of scikit-learn
    parser.add_argument("--data-home", default="")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paper = PAPERS[args.paper]
    X, y = load_dataset(paper["datasets"]["tag"], args.dataset, args.data_home or None)
    ranker = make_ranker(args.paper, args.method, args.n_jobs)
    data_params = paper["methods"][args.method].get("data_params")
    search = paper["methods"][args.method].get("search")
    transform = paper["scenarios"][args.scenario]["transform"]
    outer_cv = paper["cross_validation"]["outer"]
    # The splits of all the repetitions are generated, and those of this one kept.
    # The splitters have a fixed seed, so every method and scenario gets the same
    # splits, and the seed of the transformation of the training rankings of each
    # fold is its index among all the splits, so they also get the same rankings
    n_splits = outer_cv.get_n_splits(X, y) // n_repetitions(outer_cv)
    first = args.repetition * n_splits
    splits = list(outer_cv.split(X, y))[first : first + n_splits]
    folds = [
        run_fold(
            paper,
            ranker,
            data_params,
            search,
            transform,
            X,
            y,
            train,
            test,
            first + fold,
        )
        for fold, (train, test) in enumerate(splits)
    ]

    result = {
        "paper": args.paper,
        "dataset": args.dataset,
        "method": args.method,
        "scenario": args.scenario,
        "metric": paper["metric"],
        "repetition": args.repetition,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_labels": y.shape[1],
        "params": ranker.get_params(),
        "search": describe_search(search),
        "outer_cv": repr(outer_cv),
        "inner_cv": repr(paper["cross_validation"]["inner"]),
        "machine": describe(),
        "versions": {
            "sklr": sklr.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "folds": folds,
    }
    # Written under another name and renamed, so that the output exists only when
    # it is complete
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial")
    partial.write_text(json.dumps(result, indent=2, default=str))
    os.replace(partial, args.output)


if __name__ == "__main__":
    main()
