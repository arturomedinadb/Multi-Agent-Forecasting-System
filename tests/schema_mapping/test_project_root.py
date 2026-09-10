"""
Tests that PROJECT_ROOT, computed via Path(__file__).resolve().parents[N]
in a few modules, actually resolves to the repository root.
"""


def test_run_workflow_project_root_is_repo_root():
    from schema_mapping.run_workflow import PROJECT_ROOT

    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert (PROJECT_ROOT / "data").is_dir()
    assert (PROJECT_ROOT / "src" / "schema_mapping").is_dir()


def test_schema_mapping_prompts_factory_project_root_is_repo_root():
    from schema_mapping.prompts.factory import PROJECT_ROOT

    assert (PROJECT_ROOT / "prompts" / "schema_mapping" / "registry.yaml").exists()


def test_demand_forecasting_prompts_factory_project_root_is_repo_root():
    from ai_forecasting_agents.demand_forecasting.prompts.factory import PROJECT_ROOT

    assert (PROJECT_ROOT / "prompts" / "demand_forecasting" / "registry.yaml").exists()
