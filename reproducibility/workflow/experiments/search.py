"""Searches of the parameters of the methods by the inner cross-validation.

A search receives the ranker, the inner scorer and splitter, and the samples and
rankings of the training fold, and returns the chosen parameters and the mean inner
score of each candidate tried, as a list of dictionaries with its parameters and
its score. Its own options are fixed with :func:`functools.partial` in the
protocol of each paper.
"""

from functools import partial

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score


def grid_search(ranker, scoring, cv, X, y, *, grid):
    """Choose the parameters in a grid, all their combinations tried."""
    search = GridSearchCV(ranker, grid, scoring=scoring, cv=cv, refit=False)
    search.fit(X, y)
    results = search.cv_results_
    tried = [
        {"params": params, "score": float(score)}
        for params, score in zip(
            results["params"], results["mean_test_score"], strict=True
        )
    ]
    return search.best_params_, tried


def inner_train_size(cv, X, y):
    """Count the training samples of the smallest fold of the inner splitter."""
    return min(len(train) for train, _ in cv.split(X, y))


def doubling_search(ranker, scoring, cv, X, y, *, param, start, upper=None):
    """Choose an integer parameter by doubling it and then by a binary search.

    Starting from `start`, the parameter is doubled while the mean inner score
    improves. Then, a binary search halves the range between the best value and the
    one that stopped the doubling, keeping the upper half if its midpoint improves
    the best score and the lower half otherwise, until no value is left between
    both ends. `upper`, if given, receives the inner splitter and the samples and
    rankings, and returns the largest value allowed, which is tried instead of a
    larger one.
    """
    largest = None if upper is None else upper(cv, X, y)
    scores = {}

    def score(value):
        if value not in scores:
            candidate = clone(ranker).set_params(**{param: value})
            scores[value] = float(
                cross_val_score(candidate, X, y, scoring=scoring, cv=cv).mean()
            )
        return scores[value]

    best = start if largest is None else min(start, largest)
    stop = best
    while largest is None or best < largest:
        stop = 2 * best if largest is None else min(2 * best, largest)
        if score(best) >= score(stop):
            break
        best = stop
    while stop - best > 1:
        middle = (best + stop) // 2
        if score(middle) > score(best):
            best = middle
        else:
            stop = middle
    tried = [{"params": {param: value}, "score": s} for value, s in scores.items()]
    return {param: best}, tried


def describe(search):
    """Describe a search for the results, by its name and its options."""
    if search is None:
        return None
    if isinstance(search, partial):
        options = {
            name: getattr(value, "__name__", value)
            for name, value in search.keywords.items()
        }
        return {"name": search.func.__name__, **options}
    return {"name": search.__name__}
