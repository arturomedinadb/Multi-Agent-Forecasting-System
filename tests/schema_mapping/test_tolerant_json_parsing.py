"""
Tests for _parse_json_tolerant / _extract_first_json_tolerant: parsing JSON
tool arguments that may contain a Windows path, in either escaping style an
LLM might produce -
  - a single raw backslash (technically invalid JSON on its own), or
  - a properly double-escaped backslash (valid JSON as-is)
without corrupting the second style into doubled slashes.
"""
import json

from schema_mapping.functions import _parse_json_tolerant, _extract_first_json_tolerant


BS = chr(92)  # one backslash character, spelled out to keep test text unambiguous


class TestParseJsonTolerant:
    def test_repairs_a_single_raw_backslash(self):
        """A raw single backslash in a path makes the JSON technically
        invalid; this must still recover a clean, single-forward-slash path
        instead of raising."""
        text = '{"output_path":"C:' + BS + 'Users' + BS + 'file.csv"}'
        result = _parse_json_tolerant(text)
        assert result["output_path"] == "C:/Users/file.csv"

    def test_leaves_a_properly_escaped_backslash_untouched(self):
        """Valid JSON (a real backslash correctly double-escaped) must
        parse on the first attempt and keep its real backslash - not get
        doubled into '//' by a blind fallback replace."""
        text = '{"output_path":"C:' + BS * 2 + 'Users' + BS * 2 + 'file.csv"}'
        result = _parse_json_tolerant(text)
        assert result["output_path"] == "C:" + BS + "Users" + BS + "file.csv"

    def test_forward_slash_paths_are_unaffected(self):
        text = '{"output_path":"C:/Users/file.csv"}'
        result = _parse_json_tolerant(text)
        assert result["output_path"] == "C:/Users/file.csv"


class TestExtractFirstJsonTolerant:
    def test_repairs_a_single_raw_backslash_with_extra_data(self):
        """Same repair, via the _extract_first_json-based path used by
        callers that also need to tolerate extra trailing data."""
        text = '{"output_path":"C:' + BS + 'Users' + BS + 'file.csv"}' + " trailing junk"
        result = _extract_first_json_tolerant(text)
        assert result["output_path"] == "C:/Users/file.csv"

    def test_leaves_a_properly_escaped_backslash_untouched(self):
        text = '{"output_path":"C:' + BS * 2 + 'Users' + BS * 2 + 'file.csv"}'
        result = _extract_first_json_tolerant(text)
        assert result["output_path"] == "C:" + BS + "Users" + BS + "file.csv"
