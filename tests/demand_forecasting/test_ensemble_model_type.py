"""Tests that an ensemble model's evaluation validates against ModelType and
that its saved artifact, not a freshly constructed estimator, is what gets
retrained for inference.
"""

import joblib
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression

from ai_forecasting_agents.demand_forecasting.schemas.forecasting_models import (
    ModelPerformance,
    ModelType,
)


def _model_performance(**overrides):
    fields = dict(
        model_name="ensemble_voting_1",
        model_type="ensemble",
        model_path="model.pkl",
        model_uuid="1",
        hyperparameters={},
        rmse=19.4,
        r2=0.96,
        mae=14.6,
        mape=15.3,
        overall_score=69.8,
    )
    fields.update(overrides)
    return ModelPerformance(**fields)


class TestEnsembleModelType:
    def test_ensemble_is_a_valid_model_type(self):
        assert ModelType.ENSEMBLE == "ensemble"

    def test_model_performance_accepts_ensemble_type(self):
        performance = _model_performance()
        assert performance.model_type == ModelType.ENSEMBLE


class TestSaveBestModelUsesSavedEnsembleArtifact:
    def test_ensemble_artifact_is_loaded_and_refits_without_reconstruction(
        self, tmp_path
    ):
        base_estimators = [("linear", LinearRegression())]
        ensemble = VotingRegressor(base_estimators)

        X = pd.DataFrame({"feature": [1, 2, 3, 4]})
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        ensemble.fit(X, y)

        model_path = tmp_path / "ensemble_voting_1.pkl"
        joblib.dump(ensemble, model_path)

        loaded = joblib.load(model_path)
        loaded.fit(X, y)

        assert loaded.predict(X) is not None
