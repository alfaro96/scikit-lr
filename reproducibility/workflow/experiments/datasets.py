"""Datasets of the papers in OpenML and transformations of their rankings."""

from functools import partial

import numpy as np
from scipy.stats import rankdata
from sklearn.datasets import fetch_openml

# The datasets uploaded to OpenML by the authors of scikit-lr, with their identifiers
# under each of their tags, since a name may have several versions there
OPENML_DATASETS = {
    "label ranking": {
        "authorship": 42834,
        "bodyfat": 42837,
        "calhousing": 42839,
        "cold": 42840,
        "cpu": 42841,
        "diau": 42842,
        "dtt": 42843,
        "elevators": 42845,
        "fried": 42846,
        "glass": 42847,
        "heat": 42849,
        "housing": 42850,
        "iris": 42851,
        "pendigits": 42856,
        "segment": 42859,
        "spo": 42861,
        "stock": 42862,
        "vehicle": 42863,
        "vowel": 42865,
        "wine": 42867,
        "wisconsin": 42869,
    },
    "partial label ranking": {
        "authorship": 42835,
        "blocks": 42836,
        "breast": 42838,
        "ecoli": 42844,
        "glass": 42848,
        "letter": 42853,
        "libras": 42855,
        "pendigits": 42857,
        "satimage": 42858,
        "segment": 42860,
        "vehicle": 42864,
        "vowel": 42866,
        "yeast": 42870,
        "iris": 42871,
        "wine": 42872,
    },
}


def load_dataset(tag, name, data_home=None):
    """Load the features and the rankings of a dataset from OpenML."""
    # The liac-arff parser avoids depending on pandas
    bunch = fetch_openml(
        data_id=OPENML_DATASETS[tag][name],
        as_frame=False,
        parser="liac-arff",
        data_home=data_home,
    )
    return bunch.data, np.asarray(bunch.target, dtype=np.float64)


def remove_labels(y, random_state, *, probability):
    """Remove each label of each ranking with a probability.

    The positions of the labels left are made dense from one again, keeping their
    ties. The rankings may be left with fewer than two labels, which are kept.
    """
    y = y.copy()
    y[random_state.rand(*y.shape) < probability] = np.nan
    return rankdata(y, axis=1, method="dense", nan_policy="omit")


def missing_labels_scenarios(probabilities):
    """Build the scenarios that remove labels from the training rankings.

    There is a scenario for each probability of removing each label, named by its
    percentage, except a probability of zero, which keeps the complete training
    rankings.
    """
    scenarios = {}
    for probability in probabilities:
        if probability == 0:
            scenarios["complete"] = {
                "description": "complete training rankings",
                "transform": None,
            }
            continue
        percentage = round(100 * probability)
        scenarios[f"missing{percentage}"] = {
            "description": f"{percentage}% of the training labels removed",
            "transform": partial(remove_labels, probability=probability),
        }
    return scenarios
