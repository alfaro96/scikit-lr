import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import check_grad, minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state

from sklr.reduction._ordinal import OrdinalLogisticRegression, _loss_grad
from sklr.utils._estimator_checks import parametrize_with_checks


@parametrize_with_checks(
    [OrdinalLogisticRegression(), OrdinalLogisticRegression(C=np.inf)]
)
def test_estimators(estimator, check):
    check(estimator)


def _naive_proba(X, coef, thresholds):
    """Compute the probability of each class as a difference of cumulative logits."""
    cumulative = [np.zeros(len(X))]
    cumulative += [expit(threshold + X @ coef) for threshold in thresholds]
    cumulative.append(np.ones(len(X)))
    return np.column_stack(
        [cumulative[r + 1] - cumulative[r] for r in range(len(thresholds) + 1)]
    )


def _naive_loss(X, coef, thresholds, likelihood, sample_weight, C):
    """Compute the penalized mean negative log-likelihood sample by sample."""
    proba = _naive_proba(X, coef, thresholds)
    log_likelihood = sum(
        weight * np.log(row @ p)
        for weight, row, p in zip(sample_weight, likelihood, proba)
    )
    return (-log_likelihood + 0.5 * coef @ coef / C) / np.sum(sample_weight)


def _random_problem(n_samples, n_features, n_classes, *, incomplete, random_state):
    rng = check_random_state(random_state)
    X = rng.normal(size=(n_samples, n_features))
    classes = rng.randint(n_classes, size=n_samples)
    likelihood = np.zeros((n_samples, n_classes))
    likelihood[np.arange(n_samples), classes] = 1.0
    if incomplete:
        # Half of the samples have a random likelihood of each class, which is zero
        # for one class
        likelihood[::2] = rng.rand(n_samples // 2 + n_samples % 2, n_classes)
        likelihood[::2, rng.randint(n_classes)] = 0.0
    sample_weight = rng.uniform(0.5, 2.0, size=n_samples)
    return X, likelihood, sample_weight


def _params(coef, thresholds):
    return np.concatenate((coef, thresholds[:1], np.diff(thresholds)))


@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("incomplete", [False, True])
def test_loss_grad(n_classes, incomplete):
    """The loss matches the formula and its gradient the finite differences."""
    X, likelihood, sample_weight = _random_problem(
        20, 3, n_classes, incomplete=incomplete, random_state=0
    )
    C = 0.7
    rng = np.random.RandomState(1)
    coef = rng.normal(size=3)
    thresholds = np.sort(rng.normal(size=n_classes - 1))
    params = _params(coef, thresholds)
    with np.errstate(divide="ignore"):
        log_likelihood = np.log(likelihood)
    args = (
        X,
        likelihood,
        log_likelihood,
        sample_weight / sample_weight.sum(),
        1 / (C * sample_weight.sum()),
    )
    loss, _ = _loss_grad(params, *args)
    expected = _naive_loss(X, coef, thresholds, likelihood, sample_weight, C)
    assert_allclose(loss, expected, rtol=1e-12)
    error = check_grad(
        lambda p: _loss_grad(p, *args)[0], lambda p: _loss_grad(p, *args)[1], params
    )
    # The finite differences of check_grad have an error of about the square
    # root of the machine precision
    assert error < 1e-6


@pytest.mark.parametrize("C", [0.1, 1.0, 100.0])
def test_two_classes_is_logistic_regression(C):
    """With two classes, the model is the binary logistic regression."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(50, 3))
    y = (X @ [1.0, -2.0, 0.5] + rng.normal(size=50) > 0).astype(int)
    sample_weight = rng.uniform(0.5, 2.0, size=50)
    model = OrdinalLogisticRegression(C=C, tol=1e-10, max_iter=1000)
    model.fit(X, y, sample_weight)
    expected = LogisticRegression(C=C, tol=1e-10, max_iter=1000)
    expected.fit(X, y, sample_weight)
    # P(y <= 0) = expit(threshold + coef @ x) = 1 - expit(coef_ @ x + intercept_)
    assert_allclose(model.coef_, -expected.coef_[0], rtol=1e-6)
    assert_allclose(model.thresholds_, -expected.intercept_, rtol=1e-6)
    assert_allclose(model.predict_proba(X), expected.predict_proba(X), atol=1e-8)


@pytest.mark.parametrize("n_classes", [3, 4])
@pytest.mark.parametrize("incomplete", [False, True])
def test_fit_maximizes_likelihood(n_classes, incomplete):
    """The fit reaches the optimum found by a naive optimizer of the objective."""
    X, likelihood, sample_weight = _random_problem(
        40, 2, n_classes, incomplete=incomplete, random_state=0
    )
    model = OrdinalLogisticRegression(tol=1e-10, max_iter=1000)
    model.fit_likelihood(X, likelihood, np.arange(n_classes), sample_weight)

    def objective(params):
        # The increments are kept positive by optimizing their logarithm
        coef = params[:2]
        thresholds = params[2] + np.concatenate(([0], np.cumsum(np.exp(params[3:]))))
        return _naive_loss(X, coef, thresholds, likelihood, sample_weight, 1.0)

    naive = minimize(
        objective,
        np.zeros(n_classes + 1),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 100000, "maxfev": 100000},
    )
    loss = _naive_loss(
        X, model.coef_, model.thresholds_, likelihood, sample_weight, 1.0
    )
    assert loss <= naive.fun + 1e-9
    if not incomplete:
        # The objective is convex with complete data, so the optimum is unique
        coef = naive.x[:2]
        thresholds = naive.x[2] + np.concatenate(([0], np.cumsum(np.exp(naive.x[3:]))))
        assert_allclose(model.coef_, coef, atol=1e-4)
        assert_allclose(model.thresholds_, thresholds, atol=1e-4)


def test_predict_proba():
    """The probabilities of each sample sum to one and the thresholds are sorted."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 2))
    y = np.digitize(X @ [1.0, 1.0] + rng.normal(size=60), [-1, 0, 1])
    model = OrdinalLogisticRegression().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (60, 4)
    assert np.all(proba >= 0)
    assert_allclose(proba.sum(axis=1), 1.0)
    assert np.all(np.diff(model.thresholds_) >= 0)
    assert np.mean(model.predict(X) == y) > 0.5


def test_uninformative_likelihood():
    """A likelihood that is the same for all the classes keeps them equally likely."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(10, 2))
    model = OrdinalLogisticRegression().fit_likelihood(
        X, np.full((10, 4), 0.3), np.arange(1, 5)
    )
    assert_allclose(model.predict_proba(X), 0.25)


def test_unobserved_class():
    """A class between two observed ones that is never observed gets no probability."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 1))
    y = np.where(X[:, 0] > 0, 2, 0)
    model = OrdinalLogisticRegression().fit_likelihood(X, np.eye(3)[y], np.arange(3))
    assert_allclose(model.predict_proba(X)[:, 1], 0.0, atol=1e-6)
