import os
import sys
from importlib.abc import MetaPathFinder

os.environ["ABACUSAGENT_MODEL"] = "test"


def test_dos_module_imports_with_available_abacustest_dos_api():
    from abacusagent.modules.submodules.dos import DOSData, PDOSData

    assert DOSData is not None
    assert PDOSData is not None


def test_regex_is_available_for_abacustest_eos_import_path():
    import regex
    from abacustest.lib_model.comm_eos import eos_fit

    assert regex is not None
    assert callable(eos_fit)


def test_adam_runtime_submodules_do_not_require_mcp_import(monkeypatch):
    class BlockMcpFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "mcp" or fullname.startswith("mcp."):
                raise ModuleNotFoundError("blocked mcp import for runtime compatibility test")
            return None

    finder = BlockMcpFinder()
    monkeypatch.delenv("ABACUSAGENT_MODEL", raising=False)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])

    from abacusagent.modules.submodules.abacus import abacus_prepare
    from abacusagent.modules.submodules.phonon import abacus_phonon_dispersion
    from abacusagent.modules.submodules.pyatb import pyatb_calculate_jdos

    assert callable(abacus_prepare)
    assert callable(abacus_phonon_dispersion)
    assert callable(pyatb_calculate_jdos)


def test_generate_input_chern_compatible_with_pyatb_new_signature(monkeypatch):
    from abacusagent.modules.util import pyatb as pyatb_module

    calls = []

    def fake_generate_input_chern(input_text, n_occu, occu_switch, dim, lattice_vectors, method, mp_density):
        calls.append((input_text, n_occu, occu_switch, dim, lattice_vectors, method, mp_density))
        return "chern-input"

    monkeypatch.setattr(pyatb_module, "generate_input_chern", fake_generate_input_chern)

    assert pyatb_module.generate_input_chern_compatible(
        "input-text",
        4,
        0,
        "3",
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "direct",
        0.05,
    ) == "chern-input"
    assert calls == [
        (
            "input-text",
            4,
            0,
            "3",
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "direct",
            0.05,
        )
    ]


def test_generate_input_chern_compatible_with_pyatb_old_signature(monkeypatch):
    from abacusagent.modules.util import pyatb as pyatb_module

    calls = []

    def fake_generate_input_chern(input_text, n_occu, occu_switch, dim, lattice_vectors, method):
        calls.append((input_text, n_occu, occu_switch, dim, lattice_vectors, method))
        return "chern-input"

    monkeypatch.setattr(pyatb_module, "generate_input_chern", fake_generate_input_chern)

    assert pyatb_module.generate_input_chern_compatible(
        "input-text",
        4,
        0,
        "3",
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "direct",
        0.05,
    ) == "chern-input"
    assert calls == [
        (
            "input-text",
            4,
            0,
            "3",
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "direct",
        )
    ]
