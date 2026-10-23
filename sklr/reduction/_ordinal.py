"""Ordinal logistic regression, the default estimator of the labelwise ranker."""

from numbers import Integral, Real

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_expit, logit, logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from sklr.utils._sklearn_compat import (
    Interval,
    _check_optimize_result,
    _check_sample_weight,
    _fit_context,
)


def _log_proba(X, coef, thresholds, increments):
    """Compute the logarithm of the probability of each class of each sample.

    The `increments` between consecutive `thresholds` are given apart, because
    the thresholds may lose them to rounding.
    """
    z = thresholds + (X @ coef)[:, np.newaxis]
    log_proba = np.empty((X.shape[0], len(thresholds) + 1))
    log_proba[:, 0] = log_expit(z[:, 0])
    log_proba[:, -1] = log_expit(-z[:, -1])
    # The difference of two consecutive cumulative probabilities, written as a
    # product so that it does not cancel out when both are close to one
    log_gaps = np.log(-np.expm1(-increments))
    log_proba[:, 1:-1] = log_expit(z[:, 1:]) + log_expit(-z[:, :-1]) + log_gaps
    return log_proba, z


def _unpack(params, n_features):
    """Split the parameters into the coefficients and the thresholds."""
    coef = params[:n_features]
    thresholds = params[n_features] + np.concatenate(
        ([0.0], np.cumsum(params[n_features + 1 :]))
    )
    return coef, thresholds


def _loss_grad(params, X, likelihood, log_likelihood, sample_weight, l2_reg_strength):
    """Compute the penalized negative log-likelihood and its gradient."""
    n_features = X.shape[1]
    coef, thresholds = _unpack(params, n_features)
    log_proba, z = _log_proba(X, coef, thresholds, params[n_features + 1 :])
    log_mixture = logsumexp(log_proba + log_likelihood, axis=1)
    loss = -sample_weight @ log_mixture + 0.5 * l2_reg_strength * coef @ coef
    # Each threshold only moves the cumulative probability of its class, which
    # adds to the probability of its class and subtracts from that of the next one
    derivative = np.exp(log_expit(z) + log_expit(-z) - log_mixture[:, np.newaxis])
    grad_z = -sample_weight[:, np.newaxis] * (
        (likelihood[:, :-1] - likelihood[:, 1:]) * derivative
    )
    grad_thresholds = grad_z.sum(axis=0)
    grad = np.empty_like(params)
    grad[:n_features] = X.T @ grad_z.sum(axis=1) + l2_reg_strength * coef
    grad[n_features] = grad_thresholds.sum()
    grad[n_features + 1 :] = np.cumsum(grad_thresholds[::-1])[::-1][1:]
    return loss, grad


class OrdinalLogisticRegression(ClassifierMixin, BaseEstimator):
    """Ordinal logistic regression with the cumulative logit model.

    The logit of the probability that the class of a sample is at most the
    :math:`r`-th class is :math:`\\beta_r + \\boldsymbol{w}^\\top \\boldsymbol{x}`,
    with non-decreasing thresholds :math:`\\beta_r`, as in equation (20) of
    [cheng_labelwise_2013], and the classes are ordered as in ``classes_``. The
    coefficients are penalized as in :class:`~sklearn.linear_model.LogisticRegression`,
    and the thresholds are not. Besides the classes of the samples, it learns from
    the likelihood of what is observed of each sample given each class with
    ``fit_likelihood``, which the labelwise label ranker uses for the incomplete
    rankings.
    """

    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(self, *, alpha=1.0, tol=1e-4, max_iter=100):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # The classes of the common checks are not ordered, so an ordinal model
        # cannot separate them well
        tags.classifier_tags.poor_score = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the model to the classes of the samples."""
        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=np.float64, ensure_non_negative=True
        )
        classes, y = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError(
                "Expected samples of at least 2 classes in y, got 1 class: "
                f"{classes[0]!r}."
            )
        likelihood = np.zeros((len(y), len(classes)))
        likelihood[np.arange(len(y)), y] = 1.0
        return self._fit(X, likelihood, classes, sample_weight)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_likelihood(self, X, likelihood, classes, sample_weight=None):
        """Fit the model to the likelihood of each sample given each class.

        It is the protocol of the estimators of the labelwise label ranker that
        learn from incomplete rankings, so its arguments are not validated.
        """
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        return self._fit(X, likelihood, classes, sample_weight)

    def _fit(self, X, likelihood, classes, sample_weight):
        n_features = X.shape[1]
        n_classes = len(classes)
        self.classes_ = classes
        # The model starts from equally likely classes, which it keeps for the
        # samples whose likelihood is the same for all the classes
        thresholds = logit(np.arange(1, n_classes) / n_classes)
        params = np.concatenate((np.zeros(n_features + 1), np.diff(thresholds)))
        params[n_features] = thresholds[0]
        # As in LogisticRegression, the loss is the mean over the samples, so the
        # penalty is divided by their total weight
        sum_weight = sample_weight.sum()
        l2_reg_strength = self.alpha / sum_weight
        with np.errstate(divide="ignore"):
            log_likelihood = np.log(likelihood)
        result = minimize(
            _loss_grad,
            params,
            method="L-BFGS-B",
            jac=True,
            args=(
                X,
                likelihood,
                log_likelihood,
                sample_weight / sum_weight,
                l2_reg_strength,
            ),
            # The increments between consecutive thresholds keep them ordered. They
            # stay above zero because the solver may try a step on the bound, where
            # an observed class would have a zero probability and an infinite loss
            bounds=[(None, None)] * (n_features + 1)
            + [(np.finfo(float).eps, None)] * (n_classes - 2),
            options={
                "maxiter": self.max_iter,
                "maxls": 50,
                "gtol": self.tol,
                "ftol": 64 * np.finfo(float).eps,
            },
        )
        self.n_iter_ = _check_optimize_result("lbfgs", result, self.max_iter)
        self.coef_, self.thresholds_ = _unpack(result.x, n_features)
        return self

    def predict_proba(self, X):
        """Predict the probability of each class of each sample."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        # Two equal thresholds give the class between them a zero probability
        with np.errstate(divide="ignore"):
            log_proba, _ = _log_proba(
                X, self.coef_, self.thresholds_, np.diff(self.thresholds_)
            )
        return np.exp(log_proba)

    def predict(self, X):
        """Predict the most probable class of each sample."""
        check_is_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
