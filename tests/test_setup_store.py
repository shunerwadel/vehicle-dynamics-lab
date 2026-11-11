"""Tests for saving/loading setups in ``utils/setup_store.py``."""

from __future__ import annotations

import json

import pytest

from physics import VehicleSetup
from utils.agent_parser import setup_to_dict
from utils.setup_store import (
    BASELINE_PATH_ENV,
    DEFAULT_BASELINE_PATH,
    SetupStoreError,
    baseline_path,
    delete_setup,
    load_setup,
    save_setup,
)


@pytest.fixture
def template() -> dict:
    return setup_to_dict(VehicleSetup())


@pytest.fixture
def path(tmp_path):
    return tmp_path / "nested" / "baseline.json"


def test_missing_file_returns_none(template, path):
    assert load_setup(template, path) is None


def test_round_trip(template, path):
    setup = json.loads(json.dumps(template))
    setup["front_suspension"]["spring_rate"] = 72_000.0
    saved = save_setup(setup, path)
    loaded = load_setup(template, path)
    assert loaded.setup == setup
    assert loaded.saved_at == saved.saved_at
    assert not path.with_suffix(".json.tmp").exists()


def test_name_round_trip_and_cleanup(template, path):
    save_setup(template, path, name="  Stiff   front ARB  ")
    assert load_setup(template, path).name == "Stiff front ARB"
    save_setup(template, path, name="   ")
    assert load_setup(template, path).name is None


def test_file_without_name_still_loads(template, path):
    save_setup(template, path, name="x")
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["name"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_setup(template, path).name is None


def test_save_overwrites_previous(template, path):
    save_setup(template, path)
    changed = json.loads(json.dumps(template))
    changed["mass"] = 1300.0
    save_setup(changed, path)
    assert load_setup(template, path).setup["mass"] == 1300.0


def test_delete_is_idempotent(template, path):
    save_setup(template, path)
    delete_setup(path)
    delete_setup(path)
    assert load_setup(template, path) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("mass"),
        lambda s: s.update(new_parameter=1.0),
        lambda s: s["front_suspension"].update(spring_rate="stiff"),
        lambda s: s.update(aero=5.0),
    ],
)
def test_layout_mismatch_rejected(template, path, mutate):
    setup = json.loads(json.dumps(template))
    mutate(setup)
    save_setup(setup, path)
    with pytest.raises(SetupStoreError):
        load_setup(template, path)


def test_env_var_overrides_default_location(template, path, monkeypatch):
    monkeypatch.setenv(BASELINE_PATH_ENV, str(path))
    assert baseline_path() == path
    save_setup(template, name="From env")
    assert load_setup(template).name == "From env"
    delete_setup()
    assert not path.exists()
    monkeypatch.delenv(BASELINE_PATH_ENV)
    assert baseline_path() == DEFAULT_BASELINE_PATH


def test_corrupt_or_wrong_version_rejected(template, path):
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SetupStoreError):
        load_setup(template, path)
    path.write_text(json.dumps({"format_version": 99, "setup": template}), encoding="utf-8")
    with pytest.raises(SetupStoreError):
        load_setup(template, path)
