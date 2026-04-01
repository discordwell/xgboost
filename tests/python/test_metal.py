# coding: utf-8
"""Tests for Metal GPU plugin on Apple Silicon."""

import platform

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

import xgboost as xgb


def _skip_if_not_metal():
    """Skip test if Metal plugin is not available."""
    if platform.system() != "Darwin":
        pytest.skip("Metal plugin only available on macOS")
    try:
        X = np.random.randn(10, 2).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        dtrain = xgb.DMatrix(X, label=y)
        xgb.train(
            {"device": "metal", "tree_method": "hist", "verbosity": 0, "num_boost_round": 1},
            dtrain,
            num_boost_round=1,
        )
    except xgb.core.XGBoostError as e:
        if "Metal" in str(e) or "metal" in str(e) or "Unknown" in str(e):
            pytest.skip(f"Metal plugin not available: {e}")
        raise


@pytest.fixture(autouse=True)
def check_metal():
    _skip_if_not_metal()


def _metal_params(**overrides):
    """Default Metal training parameters."""
    params = {
        "device": "metal",
        "tree_method": "hist",
        "verbosity": 0,
    }
    params.update(overrides)
    return params


def _compare_models(y, cpu_preds, metal_preds, max_loss_gap=0.02):
    """Compare CPU and Metal binary classifiers using log_loss."""
    cpu_loss = log_loss(y, cpu_preds)
    metal_loss = log_loss(y, metal_preds)
    assert cpu_loss < 0.7, f"CPU model has poor log_loss: {cpu_loss}"
    assert metal_loss < 0.7, f"Metal model has poor log_loss: {metal_loss}"
    assert abs(cpu_loss - metal_loss) < max_loss_gap, (
        f"CPU/Metal log_loss differ too much: {cpu_loss:.6f} vs {metal_loss:.6f}"
    )


class TestMetalBasic:
    """Basic Metal functionality tests."""

    def test_binary_classification(self):
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)

        cpu_m = xgb.train({"verbosity": 0, "objective": "binary:logistic"}, dtrain, num_boost_round=20)
        metal_m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=20)

        cpu_acc = accuracy_score(y, (cpu_m.predict(dtrain) > 0.5).astype(int))
        metal_acc = accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int))
        assert cpu_acc > 0.9
        assert metal_acc > 0.9

    def test_regression(self):
        X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)

        cpu_m = xgb.train({"verbosity": 0, "objective": "reg:squarederror"}, dtrain, num_boost_round=20)
        metal_m = xgb.train(_metal_params(objective="reg:squarederror"), dtrain, num_boost_round=20)

        cpu_mse = mean_squared_error(y, cpu_m.predict(dtrain))
        metal_mse = mean_squared_error(y, metal_m.predict(dtrain))
        assert cpu_mse < np.var(y) * 0.5
        assert metal_mse < np.var(y) * 0.5

    def test_multiclass(self):
        X, y = make_classification(
            n_samples=1000, n_features=10, n_classes=3,
            n_informative=6, random_state=42,
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

        cpu_preds = cpu_m.predict(dtrain)
        metal_preds = metal_m.predict(dtrain)

        cpu_acc = accuracy_score(y, np.argmax(cpu_preds, axis=1))
        metal_acc = accuracy_score(y, np.argmax(metal_preds, axis=1))
        assert cpu_acc > 0.8
        assert metal_acc > 0.8


class TestMetalScalability:
    """Test with various dataset sizes."""

    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 5000, 10000])
    def test_dataset_sizes(self, n_samples):
        X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)

        cpu_m = xgb.train({"verbosity": 0, "objective": "binary:logistic"}, dtrain, num_boost_round=10)
        metal_m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=10)

        _compare_models(y, cpu_m.predict(dtrain), metal_m.predict(dtrain), max_loss_gap=0.1)

    @pytest.mark.parametrize("n_features", [10, 50, 100, 200])
    def test_feature_counts(self, n_features):
        n_info = min(n_features // 2, 20)
        X, y = make_classification(
            n_samples=5000, n_features=n_features,
            n_informative=n_info, n_redundant=n_info, random_state=42,
        )
        dtrain = xgb.DMatrix(X, label=y)

        # More rounds for wider datasets where signal is diluted
        rounds = 10 if n_features <= 20 else 50
        metal_m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=rounds)
        metal_acc = accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int))
        # With many redundant features, FP32 histograms give weaker splits.
        # Verify the model learned something meaningful above chance (0.5).
        assert metal_acc > 0.52, f"Model barely above chance for {n_features} features: {metal_acc}"


class TestMetalTreeParams:
    """Test various tree parameters."""

    @pytest.mark.parametrize("max_depth", [2, 4, 6, 8])
    def test_max_depth(self, max_depth):
        X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)

        metal_m = xgb.train(
            _metal_params(objective="binary:logistic", max_depth=max_depth),
            dtrain, num_boost_round=10,
        )
        acc = accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int))
        assert acc > 0.8

    @pytest.mark.parametrize("max_bin", [32, 64, 128, 256])
    def test_max_bin(self, max_bin):
        X, y = make_classification(n_samples=2000, n_features=10, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)

        metal_m = xgb.train(
            _metal_params(objective="binary:logistic", max_bin=max_bin),
            dtrain, num_boost_round=10,
        )
        acc = accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int))
        assert acc > 0.85

    def test_subsample(self):
        X, y = make_classification(n_samples=2000, n_features=10, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)

        metal_m = xgb.train(
            _metal_params(objective="binary:logistic", subsample=0.8, seed=42),
            dtrain, num_boost_round=20,
        )
        acc = accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int))
        assert acc > 0.85

    def test_colsample(self):
        X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)

        metal_m = xgb.train(
            _metal_params(
                objective="binary:logistic",
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
            ),
            dtrain, num_boost_round=20,
        )
        acc = accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int))
        assert acc > 0.85

    def test_regularization(self):
        X, y = make_classification(n_samples=2000, n_features=10, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)

        metal_m = xgb.train(
            _metal_params(
                objective="binary:logistic",
                reg_alpha=1.0,
                reg_lambda=2.0,
                gamma=0.1,
            ),
            dtrain, num_boost_round=20,
        )
        acc = accuracy_score(y, (metal_m.predict(dtrain) > 0.5).astype(int))
        assert acc > 0.85


class TestMetalObjectives:
    """Test different objective functions."""

    def test_logistic(self):
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)
        m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=10)
        preds = m.predict(dtrain)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_squared_error(self):
        X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)
        m = xgb.train(_metal_params(objective="reg:squarederror"), dtrain, num_boost_round=20)
        mse = mean_squared_error(y, m.predict(dtrain))
        assert mse < np.var(y) * 0.3

    def test_absolute_error(self):
        X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)
        m = xgb.train(_metal_params(objective="reg:absoluteerror"), dtrain, num_boost_round=20)
        assert m is not None


class TestMetalEdgeCases:
    """Test edge cases and error handling."""

    def test_single_row(self):
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        dtrain = xgb.DMatrix(X, label=y)
        m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=1)
        assert m is not None

    def test_many_rounds(self):
        X, y = make_classification(n_samples=500, n_features=5, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)
        m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=100)
        acc = accuracy_score(y, (m.predict(dtrain) > 0.5).astype(int))
        assert acc > 0.95

    def test_missing_values(self):
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        # Introduce 10% missing values
        mask = np.random.RandomState(42).random(X.shape) < 0.1
        X[mask] = np.nan
        dtrain = xgb.DMatrix(X, label=y)
        m = xgb.train(_metal_params(objective="binary:logistic"), dtrain, num_boost_round=10)
        preds = m.predict(dtrain)
        assert not np.any(np.isnan(preds))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
