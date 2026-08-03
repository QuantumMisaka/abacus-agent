"""Unit tests for phonon submodule helpers (no ABACUS execution required)."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from abacusagent.modules.submodules.phonon import (
    _json_safe_band_structure,
    _ndarray_to_list,
    initialize_phonopy_calc,
)
from abacustest.lib_prepare.abacus import AbacusStru

_TEST_DATA = Path(__file__).parent / "abacus_inputs_dirs"


def test_initialize_phonopy_calc_emits_no_primitive_auto_warning():
    """Non-primitive input cells must not trigger phonopy PrimitiveMatrixAutoDefaultWarning."""
    stru = AbacusStru.ReadStru(_TEST_DATA / "gamma-TiAl-P4mmm" / "STRU")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        phonon = initialize_phonopy_calc(stru, supercell=[2, 2, 1])
    primitive_warnings = [
        item for item in caught
        if "PrimitiveMatrixAutoDefaultWarning" in type(item.message).__name__
        or "primitive" in str(item.message).lower()
    ]
    assert primitive_warnings == []
    assert phonon.primitive_matrix is not None


def test_json_safe_band_structure_converts_numpy_arrays():
    class FakePhonon:
        band_structure_dict = {
            "frequencies": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "distances": np.array([0.0, 0.1]),
            "labels": ["G", "X"],
        }

    payload = _json_safe_band_structure(FakePhonon())
    # round-trip must succeed and keep values
    assert json.loads(json.dumps(payload)) == payload
    assert payload["frequencies"] == [[1.0, 2.0], [3.0, 4.0]]
    assert payload["labels"] == ["G", "X"]


def test_json_safe_band_structure_handles_missing_data():
    class EmptyPhonon:
        band_structure_dict = None

    assert _json_safe_band_structure(EmptyPhonon()) == {}


def test_ndarray_to_list_converts_and_rejects_other_types():
    assert _ndarray_to_list(np.array([1, 2])) == [1, 2]
    with pytest.raises(TypeError):
        _ndarray_to_list({"a": 1})
