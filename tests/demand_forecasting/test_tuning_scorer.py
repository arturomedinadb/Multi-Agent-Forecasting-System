"""
Tests for the hyperparameter tuning scorer: it must produce real scores with
the installed scikit-learn. A scorer that raises inside cross-validation is
not reported as an error - sklearn records NaN for that fold - so tuning
would silently select parameters based on nothing.
"""

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

TRAINING_FUNCTIONS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "ai_forecasting_agents"
    / "demand_forecasting"
    / "tools"
    / "training_functions.py"
)

TUNING_SCORER = "neg_root_mean_squared_error"


class TestTuningScorer:
    def test_scorer_produces_real_scores_not_nan(self):
        rng = np.random.default_rng(0)
        X = rng.random((40, 3))
        y = rng.random(40) * 100

        search = GridSearchCV(
            RandomForestRegressor(n_estimators=5, random_state=0),
            {"max_depth": [2, 3]},
            scoring=TUNING_SCORER,
            cv=TimeSeriesSplit(n_splits=2),
        )
        search.fit(X, y)

        scores = search.cv_results_["mean_test_score"]
        assert not np.isnan(scores).any()
        assert search.best_params_

    def test_removed_squared_argument_is_not_used(self):
        """mean_squared_error's squared argument was removed in
        scikit-learn 1.6; using it makes every fold score NaN."""
        source = TRAINING_FUNCTIONS.read_text(encoding="utf-8")
        assert "squared=False" not in source
