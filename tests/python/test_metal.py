# coding: utf-8
"""Tests for Metal GPU plugin on Apple Silicon.

Run with: XGBOOST_TEST_METAL=1 pytest tests/python/test_metal.py
"""

import os

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error

import xgboost as xgb

pytestmark = pytest.mark.skipif(
    os.environ.get("XGBOOST_TEST_METAL", "0") != "1",
    reason="Set XGBOOST_TEST_METAL=1 to test Metal GPU plugin.",
)


def _metal_params(**overrides):
    """Default Metal training parameters."""
    params = {"device": "metal", "tree_method": "hist", "verbosity": 0}
    params.update(overrides)
    return params


def test_binary_classification():
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    cpu_m = xgb.train({"verbosity": 0, "objective": "binary:logistic"}, dtrain, num_boost_round=20)
    metal_m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=20)
    assert accuracy_score(y, (cpu_m.predict(dtrain) > 0.5).astype(int)) > 0.9
    assert accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int)) > 0.9


def test_regression():
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    cpu_m = xgb.train({"verbosity": 0, "objective": "reg:squarederror"}, dtrain, num_boost_round=20)
    metal_m = xgb.train(_metal_params(objective="reg:squarederror"), dtrain, num_boost_round=20)
    assert mean_squared_error(y, cpu_m.predict(dtrain)) < np.var(y) * 0.5
    assert mean_squared_error(y, metal_m.predict(dtrain)) < np.var(y) * 0.5


def test_multiclass():
    X, y = make_classification(
        n_samples=1000, n_features=10, n_classes=3, n_informative=6, random_state=42
    )
    dtrain = xgb.DMatrix(X, label=y)
    cpu_m = xgb.train(
        {"verbosity": 0, "objective": "multi:softprob", "num_class": 3},
        dtrain, num_boost_round=10,
    )
    metal_m = xgb.train(
        _metal_params(objective="multi:softprob", num_class=3),
        dtrain, num_boost_round=10,
    )
    assert accuracy_score(y, np.argmax(cpu_m.predict(dtrain), axis=1)) > 0.8
    assert accuracy_score(y, np.argmax(metal_m.predict(dtrain), axis=1)) > 0.8


@pytest.mark.parametrize("n_samples", [100, 500, 1000, 5000, 10000])
def test_dataset_sizes(n_samples):
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    metal_m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=10)
    acc = accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int))
    assert acc > 0.52, f"Model barely above chance for n={n_samples}: {acc}"


@pytest.mark.parametrize("max_depth", [2, 4, 6, 8])
def test_max_depth(max_depth):
    X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    metal_m = xgb.train(
        _metal_params(objective="binary:logistic", max_depth=max_depth),
        dtrain, num_boost_round=10,
    )
    assert accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int)) > 0.8


@pytest.mark.parametrize("max_bin", [32, 64, 128, 256])
def test_max_bin(max_bin):
    X, y = make_classification(n_samples=2000, n_features=10, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    metal_m = xgb.train(
        _metal_params(objective="binary:logistic", max_bin=max_bin),
        dtrain, num_boost_round=10,
    )
    assert accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int)) > 0.85


def test_subsample():
    X, y = make_classification(n_samples=2000, n_features=10, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    metal_m = xgb.train(
        _metal_params(objective="binary:logistic", subsample=0.8, seed=42),
        dtrain, num_boost_round=20,
    )
    assert accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int)) > 0.85


def test_colsample():
    X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    metal_m = xgb.train(
        _metal_params(objective="binary:logistic", colsample_bytree=0.8, colsample_bylevel=0.8),
        dtrain, num_boost_round=20,
    )
    assert accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int)) > 0.85


def test_regularization():
    X, y = make_classification(n_samples=2000, n_features=10, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    metal_m = xgb.train(
        _metal_params(objective="binary:logistic", reg_alpha=1.0, reg_lambda=2.0, gamma=0.1),
        dtrain, num_boost_round=20,
    )
    assert accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int)) > 0.85


def test_logistic_predictions_bounded():
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=10)
    preds = m.predict(dtrain)
    assert np.all((preds >= 0) & (preds <= 1))


def test_squared_error():
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    m = xgb.train(_metal_params(objective="reg:squarederror"), dtrain, num_boost_round=20)
    assert mean_squared_error(y, m.predict(dtrain)) < np.var(y) * 0.3


def test_absolute_error():
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    m = xgb.train(_metal_params(objective="reg:absoluteerror"), dtrain, num_boost_round=20)
    assert m is not None


def test_single_row():
    X = np.array([[1.0, 2.0]])
    y = np.array([1.0])
    dtrain = xgb.DMatrix(X, label=y)
    m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=1)
    assert m is not None


def test_many_rounds():
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)
    m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=100)
    assert accuracy_score(y, (m.predict(dtrain) > 0.5).astype(int)) > 0.95


def test_missing_values():
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    mask = np.random.RandomState(42).random(X.shape) < 0.1
    X[mask] = np.nan
    dtrain = xgb.DMatrix(X, label=y)
    m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=10)
    preds = m.predict(dtrain)
    assert not np.any(np.isnan(preds))
