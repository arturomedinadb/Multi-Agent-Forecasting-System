"""
Tests for training_functions.py: train_models' error handling, and
create_model_configs' per-model-type hyperparameter assignment.

Async tools are invoked via asyncio.run() directly (not a pytest-asyncio
marker) so these tests don't depend on any pytest plugin being installed.
"""

import asyncio
import json

from ai_forecasting_agents.demand_forecasting.tools.training_functions import (
    create_model_configs,
    train_models,
)


def _invoke(tool, **kwargs):
    async def _run():
        result = tool.on_invoke_tool(None, json.dumps(kwargs))
        if hasattr(result, "__await__"):
            result = await result
        return result

    return asyncio.run(_run())


class TestTrainModels:
    def test_returns_error_dict_instead_of_raising_on_failure(self):
        """When training can't proceed (e.g. the data directory has no
        prepared splits), the tool returns a structured error dict rather
        than raising."""
        all_model_configs = {
            "configs": [
                {
                    "model_type": "xgboost",
                    "model_name": "x",
                    "model_uuid": "1",
                    "hyperparameters": {},
                }
            ],
            "total_configs": 1,
            "iteration": 1,
            "session_uuid": "1",
        }
        result = _invoke(
            train_models,
            all_model_configs=all_model_configs,
            data_dir="C:/this_directory_does_not_exist_xyz",
            output_dir="C:/this_directory_does_not_exist_xyz",
        )
        assert result["success"] is False
        assert "error" in result and result["error"]


class TestCreateModelConfigs:
    def test_each_model_type_gets_its_own_hyperparameters(self):
        hyperparameters_json = json.dumps(
            {
                "xgboost": {"max_depth": 3},
                "lightgbm": {"max_depth": 9},
            }
        )
        result = _invoke(
            create_model_configs,
            model_types=["xgboost", "lightgbm"],
            hyperparameters_json=hyperparameters_json,
            iteration=1,
        )
        by_type = {c.model_type.value: c for c in result.configs}
        assert by_type["xgboost"].hyperparameters["max_depth"] == 3
        assert by_type["lightgbm"].hyperparameters["max_depth"] == 9

    def test_falls_back_to_defaults_when_no_hyperparameters_given(self):
        result = _invoke(create_model_configs, model_types=["xgboost"], iteration=1)
        assert len(result.configs) == 1
        assert result.configs[0].hyperparameters  # defaults populated, not empty
