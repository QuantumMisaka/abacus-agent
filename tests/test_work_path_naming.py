import os
import re
import sys
from pathlib import Path

import pytest

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT / "src"))

from abacusagent.modules.util import work_path


def test_generate_work_path_uses_note_slug_without_timestamp_or_dot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(work_path, "_random_4digit_suffix", lambda: "1234")

    generated = work_path.generate_work_path(note="NiO 300K nvt")

    assert generated == "NiO-300K-nvt-1234"
    assert Path(generated).is_dir()
    assert "." not in generated


def test_generate_work_path_retries_collisions_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    suffixes = iter(["0001", "0001", "0002"])
    monkeypatch.setattr(work_path, "_random_4digit_suffix", lambda: next(suffixes))
    Path("NiO-0001").mkdir()

    generated = work_path.generate_work_path(note="NiO")

    assert generated == "NiO-0002"
    assert Path("NiO-0001").is_dir()
    assert Path("NiO-0002").is_dir()


def test_generate_work_path_create_false_returns_name_without_creating(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(work_path, "_random_4digit_suffix", lambda: "1234")

    generated = work_path.generate_work_path(note="Fe scf", create=False)

    assert generated == "Fe-scf-1234"
    assert not Path(generated).exists()


def test_generate_work_path_legacy_call_uses_calling_function(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(work_path, "_random_4digit_suffix", lambda: "1234")

    def legacy_caller():
        return work_path.generate_work_path(create=False)

    generated = legacy_caller()

    assert generated == "legacy_caller-1234"
    assert re.fullmatch(r"legacy_caller-\d{4}", generated)


def test_generate_work_path_raises_after_collision_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(work_path, "_random_4digit_suffix", lambda: "0001")
    Path("NiO-0001").mkdir()

    with pytest.raises(FileExistsError):
        work_path.generate_work_path(note="NiO")
