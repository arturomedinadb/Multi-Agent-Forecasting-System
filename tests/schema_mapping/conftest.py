# Fixtures specific to schema_mapping tests.
import pytest

import schema_mapping.functions as functions_module


@pytest.fixture(autouse=True)
def _reset_metadata_repeat_guard():
    """get_all_dataset_metadata(), generate_mapped_csvs(), and
    merge_mapped_csvs_to_target() each cache per session_id at module scope
    so a repeat call in the same real workflow run doesn't redo (or re-dump
    the full payload of) work already done, and so the evaluator gates can
    tell whether the real work already succeeded this session. Tests reuse
    the same literal session_id across cases, so these caches must be
    cleared between tests to keep them independent."""
    functions_module._metadata_already_returned.clear()
    functions_module._generate_mapped_csvs_cache.clear()
    functions_module._merge_mapped_csvs_cache.clear()
    yield
    functions_module._metadata_already_returned.clear()
    functions_module._generate_mapped_csvs_cache.clear()
    functions_module._merge_mapped_csvs_cache.clear()
