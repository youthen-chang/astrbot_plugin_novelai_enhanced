import json
from pathlib import Path

import pytest
from enhanced_test.preset_defaults import seed_presets


def test_first_install_copies_all_bundled_presets(tmp_path):
    bundled = Path(__file__).resolve().parents[1] / "presets.json"
    destination = tmp_path / "plugin_data" / "presets.json"
    assert seed_presets(destination, bundled)
    assert destination.read_bytes() == bundled.read_bytes()
    data = json.loads(destination.read_bytes())
    assert len(data["characters"]) == 1062
    assert len(data["styles"]) == 61


def test_update_preserves_user_changes_and_deletions(tmp_path):
    destination = tmp_path / "presets.json"
    original = b'{"characters":{},"styles":{"custom":"watercolor"}}'
    destination.write_bytes(original)
    assert not seed_presets(destination, tmp_path / "missing-defaults.json")
    assert destination.read_bytes() == original


def test_invalid_default_cannot_create_runtime_file(tmp_path):
    bundled = tmp_path / "defaults.json"
    bundled.write_text('{"characters": []}', encoding="utf-8")
    destination = tmp_path / "runtime" / "presets.json"
    with pytest.raises(ValueError):
        seed_presets(destination, bundled)
    assert not destination.exists()
