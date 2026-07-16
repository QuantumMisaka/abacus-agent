import os
import sys
import ast
import types
from pathlib import Path
from importlib.abc import MetaPathFinder

os.environ["ABACUSAGENT_MODEL"] = "test"
TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT / "src"))


def _install_runtime_stubs(monkeypatch):
    abacustest_pkg = types.ModuleType("abacustest")
    abacustest_pkg.__path__ = []
    lib_prepare_pkg = types.ModuleType("abacustest.lib_prepare")
    lib_prepare_pkg.__path__ = []
    abacus_mod = types.ModuleType("abacustest.lib_prepare.abacus")
    collect_pkg = types.ModuleType("abacustest.lib_collectdata")
    collect_pkg.__path__ = []
    collect_mod = types.ModuleType("abacustest.lib_collectdata.collectdata")
    pyatb_pkg = types.ModuleType("pyatb")
    pyatb_pkg.__path__ = []
    easy_use_pkg = types.ModuleType("pyatb.easy_use")
    easy_use_pkg.__path__ = []
    input_generator_mod = types.ModuleType("pyatb.easy_use.input_generator")
    stru_analyzer_mod = types.ModuleType("pyatb.easy_use.stru_analyzer")

    abacus_mod.ReadInput = lambda *args, **kwargs: {}
    abacus_mod.WriteInput = lambda *args, **kwargs: None
    abacus_mod.WriteKpt = lambda *args, **kwargs: None
    abacus_mod.AbacusStru = type("DummyAbacusStru", (), {})
    collect_mod.RESULT = object()
    input_generator_mod.generate_input_chern = lambda *args, **kwargs: "chern-input"
    stru_analyzer_mod.read_abacus_stru = lambda *args, **kwargs: None

    abacustest_pkg.lib_prepare = lib_prepare_pkg
    lib_prepare_pkg.abacus = abacus_mod
    abacustest_pkg.lib_collectdata = collect_pkg
    collect_pkg.collectdata = collect_mod
    pyatb_pkg.easy_use = easy_use_pkg
    easy_use_pkg.input_generator = input_generator_mod
    easy_use_pkg.stru_analyzer = stru_analyzer_mod

    for name, module in {
        "abacustest": abacustest_pkg,
        "abacustest.lib_prepare": lib_prepare_pkg,
        "abacustest.lib_prepare.abacus": abacus_mod,
        "abacustest.lib_collectdata": collect_pkg,
        "abacustest.lib_collectdata.collectdata": collect_mod,
        "pyatb": pyatb_pkg,
        "pyatb.easy_use": easy_use_pkg,
        "pyatb.easy_use.input_generator": input_generator_mod,
        "pyatb.easy_use.stru_analyzer": stru_analyzer_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _install_work_function_stubs(monkeypatch):
    abacustest_pkg = types.ModuleType("abacustest")
    abacustest_pkg.__path__ = []
    lib_prepare_pkg = types.ModuleType("abacustest.lib_prepare")
    lib_prepare_pkg.__path__ = []
    abacus_mod = types.ModuleType("abacustest.lib_prepare.abacus")
    collect_pkg = types.ModuleType("abacustest.lib_collectdata")
    collect_pkg.__path__ = []
    collect_mod = types.ModuleType("abacustest.lib_collectdata.collectdata")
    lib_model_pkg = types.ModuleType("abacustest.lib_model")
    lib_model_pkg.__path__ = []
    comm_mod = types.ModuleType("abacustest.lib_model.comm")
    workfunc_mod = types.ModuleType("abacustest.lib_model.model_020_workfunc")

    abacus_mod.ReadInput = lambda *args, **kwargs: {}
    collect_mod.RESULT = object()
    comm_mod.check_abacus_inputs = lambda *args, **kwargs: (True, "ok")
    workfunc_mod.prep_abacus_workfunc_calc = lambda *args, **kwargs: None
    workfunc_mod.post_workfunc_calc = lambda *args, **kwargs: ([], None, None, None)

    abacustest_pkg.lib_prepare = lib_prepare_pkg
    lib_prepare_pkg.abacus = abacus_mod
    abacustest_pkg.lib_collectdata = collect_pkg
    collect_pkg.collectdata = collect_mod
    abacustest_pkg.lib_model = lib_model_pkg
    lib_model_pkg.comm = comm_mod
    lib_model_pkg.model_020_workfunc = workfunc_mod

    for name, module in {
        "abacustest": abacustest_pkg,
        "abacustest.lib_prepare": lib_prepare_pkg,
        "abacustest.lib_prepare.abacus": abacus_mod,
        "abacustest.lib_collectdata": collect_pkg,
        "abacustest.lib_collectdata.collectdata": collect_mod,
        "abacustest.lib_model": lib_model_pkg,
        "abacustest.lib_model.comm": comm_mod,
        "abacustest.lib_model.model_020_workfunc": workfunc_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    for module_name in [
        "abacusagent.modules.work_function",
        "abacusagent.modules.submodules.work_function",
    ]:
        sys.modules.pop(module_name, None)


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


def test_jdos_submodule_accepts_and_forwards_note(monkeypatch, tmp_path):
    from abacusagent.modules.submodules import jdos_pyatb

    calls = []

    def fake_property_calculation_scf(abacus_inputs_path, mode, note=None):
        calls.append({"path": abacus_inputs_path, "mode": mode, "note": note})
        work_path = tmp_path / "jdos-work"
        (work_path / "pyatb" / "Out" / "JDOS").mkdir(parents=True)
        (work_path / "pyatb" / "Out" / "JDOS" / "JDOS.dat").write_text(
            "0.0 0.0\n1.0 1.0\n",
            encoding="utf-8",
        )
        return {"normal_end": True, "converge": True, "work_path": work_path}

    monkeypatch.setattr(jdos_pyatb, "property_calculation_scf", fake_property_calculation_scf)
    monkeypatch.setattr(jdos_pyatb.PyatbInputGenerator, "run", lambda self: None)
    monkeypatch.setattr(jdos_pyatb, "run_pyatb", lambda path: None)

    result = jdos_pyatb.pyatb_calculate_jdos(tmp_path / "inputs", note="Fe-jdos")

    assert calls == [{"path": tmp_path / "inputs", "mode": "pyatb", "note": "Fe-jdos"}]
    assert result["jdos_fig_path"].name == "jdos.png"
    assert result["jdos_data_path"] == (tmp_path / "jdos-work" / "pyatb" / "Out" / "JDOS" / "JDOS.dat").absolute()


def test_band_submodule_accepts_and_forwards_note(monkeypatch, tmp_path):
    _install_runtime_stubs(monkeypatch)
    for module_name in [
        "abacusagent.modules.submodules.band",
        "abacusagent.modules.util.pyatb",
        "abacusagent.modules.util.comm",
    ]:
        sys.modules.pop(module_name, None)

    from abacusagent.modules.submodules import band as band_module

    calls = []

    class FakeStru:
        def get_kline(self, point_number, new_stru_file, kpt_file):
            return FakeStru(), None, None, None

        def get_natoms(self):
            return 2

    class FakeAbacusStru:
        @staticmethod
        def ReadStru(path):
            return FakeStru()

    def fake_property_calculation_scf(abacus_inputs_dir, mode, always_run=False, note=None):
        calls.append(
            {
                "path": abacus_inputs_dir,
                "mode": mode,
                "always_run": always_run,
                "note": note,
            }
        )
        return {"work_path": tmp_path / "band-work", "mode": "pyatb"}

    monkeypatch.setattr(band_module, "ReadInput", lambda path: {"stru_file": "STRU"})
    monkeypatch.setattr(band_module, "AbacusStru", FakeAbacusStru)
    monkeypatch.setattr(band_module, "WriteKpt", lambda *args, **kwargs: None)
    monkeypatch.setattr(band_module, "property_calculation_scf", fake_property_calculation_scf)
    monkeypatch.setattr(
        band_module,
        "abacus_plot_band_pyatb",
        lambda work_path, energy_min, energy_max: {
            "band_gap": 1.23,
            "band_picture": tmp_path / "band.png",
        },
    )

    result = band_module.abacus_cal_band(tmp_path / "inputs", note="Si-band")

    assert calls == [
        {
            "path": tmp_path / "inputs",
            "mode": "auto",
            "always_run": False,
            "note": "Si-band",
        }
    ]
    assert result["band_gap"] == 1.23


def test_band_nscf_line_kpt_clears_inherited_gamma_selectors(monkeypatch, tmp_path):
    _install_runtime_stubs(monkeypatch)
    for module_name in [
        "abacusagent.modules.submodules.band",
        "abacusagent.modules.util.pyatb",
        "abacusagent.modules.util.comm",
    ]:
        sys.modules.pop(module_name, None)
    from abacusagent.modules.submodules import band as band_module

    work_path = tmp_path / "band-work"
    work_path.mkdir()
    inputs_path = tmp_path / "inputs"
    inputs_path.mkdir()
    written = {}
    copied = {}

    class FakeStru:
        def get_kline(self, point_number, new_stru_file, kpt_file):
            Path(kpt_file).write_text("K_POINTS\n2\nLine\n", encoding="utf-8")
            return FakeStru(), None, None, None

        def get_natoms(self):
            return 2

    class FakeAbacusStru:
        @staticmethod
        def ReadStru(path):
            return FakeStru()

    monkeypatch.setattr(band_module, "ReadInput", lambda path: {
        "stru_file": "STRU", "gamma_only": 1, "kspacing": 0.14,
        "basis_type": "pw",
    })
    monkeypatch.setattr(band_module, "AbacusStru", FakeAbacusStru)
    monkeypatch.setattr(band_module, "WriteKpt", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        band_module, "property_calculation_scf",
        lambda *args, **kwargs: {"work_path": work_path, "mode": "nscf"},
    )
    monkeypatch.setattr(band_module, "WriteInput", lambda data, path: written.update(data))
    monkeypatch.setattr(
        band_module, "copy", lambda source, target: copied.update(source=source, target=target),
        raising=False,
    )
    monkeypatch.setattr(
        band_module.shutil, "copy", lambda source, target: copied.update(source=source, target=target)
    )
    monkeypatch.setattr(band_module, "run_abacus", lambda path: None)
    monkeypatch.setattr(
        band_module, "abacus_plot_band_nscf",
        lambda *args: {"band_gap": 0.1, "band_picture": tmp_path / "band.png"},
    )

    band_module.abacus_cal_band(inputs_path, mode="nscf", note="band-line")

    assert written["calculation"] == "nscf"
    assert written["gamma_only"] == 0
    assert written["kspacing"] is None
    assert copied["source"] == str(inputs_path / "KPT_band")


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


def test_work_function_forwards_abacustest_0461_options(monkeypatch, tmp_path):
    _install_work_function_stubs(monkeypatch)

    from abacusagent.modules.submodules import work_function as work_function_module

    calls = {}

    monkeypatch.setattr(work_function_module, "check_abacus_inputs", lambda path: (True, "ok"))
    monkeypatch.setattr(work_function_module, "generate_work_path", lambda note=None: tmp_path / "work")
    monkeypatch.setattr(
        work_function_module,
        "link_abacusjob",
        lambda src, dst, copy_files, exclude_directories: calls.setdefault(
            "link",
            {
                "src": src,
                "dst": dst,
                "copy_files": copy_files,
                "exclude_directories": exclude_directories,
            },
        ),
    )

    def fake_prep(job, vacuum_dir, dipole_corr, workfunc_dir, **kwargs):
        calls["prep"] = {
            "job": job,
            "vacuum_dir": vacuum_dir,
            "dipole_corr": dipole_corr,
            "workfunc_dir": workfunc_dir,
            **kwargs,
        }
        return tmp_path / "work" / "workfunc_job"

    def fake_post(job, jobtype="abacus", vacuum_dir_specified="auto", thr=0.01):
        calls["post"] = {
            "job": job,
            "jobtype": jobtype,
            "vacuum_dir_specified": vacuum_dir_specified,
            "thr": thr,
        }
        return [{"work_function": 4.2}], tmp_path / "plot.png", tmp_path / "pot.cube", tmp_path / "profiled.dat"

    monkeypatch.setattr(work_function_module, "prep_abacus_workfunc_calc", fake_prep)
    monkeypatch.setattr(work_function_module, "run_abacus", lambda workdir: calls.setdefault("run", workdir))
    monkeypatch.setattr(work_function_module, "post_workfunc_calc", fake_post)

    result = work_function_module.abacus_cal_work_function(
        tmp_path / "inputs",
        vacuum_direction="z",
        dipole_correction=True,
        work_function_threshold=0.02,
        use_empty_atom=True,
        empty_atom_elem="Al",
        empty_atom_height=3.0,
        empty_atom_dist=2.5,
        note="Al-work-function",
    )

    assert calls["prep"]["vacuum_dir"] == "c"
    assert calls["prep"]["dipole_corr"] is True
    assert calls["prep"]["use_empty_atom"] is True
    assert calls["prep"]["empty_atom_elem"] == "Al"
    assert calls["prep"]["empty_atom_height"] == 3.0
    assert calls["prep"]["empty_atom_dist"] == 2.5
    assert calls["post"]["vacuum_dir_specified"] == "c"
    assert calls["post"]["thr"] == 0.02
    assert result["work_function_results"] == [{"work_function": 4.2}]


def test_public_work_function_entrypoint_forwards_abacustest_0461_options(monkeypatch, tmp_path):
    _install_work_function_stubs(monkeypatch)

    from abacusagent.modules import work_function as work_function_module

    calls = []

    def fake_work_function(**kwargs):
        calls.append(kwargs)
        return {"work_function_results": [{"work_function": 4.2}]}

    monkeypatch.setattr(work_function_module, "_abacus_cal_work_function", fake_work_function)

    result = work_function_module.abacus_cal_work_function(
        tmp_path / "inputs",
        vacuum_direction="auto",
        dipole_correction=True,
        work_function_threshold=0.03,
        use_empty_atom=True,
        empty_atom_elem="O",
        empty_atom_height=2.8,
        empty_atom_dist=1.9,
        note="surface-work-function",
    )

    assert calls == [
        {
            "abacus_inputs_dir": tmp_path / "inputs",
            "vacuum_direction": "auto",
            "dipole_correction": True,
            "work_function_threshold": 0.03,
            "use_empty_atom": True,
            "empty_atom_elem": "O",
            "empty_atom_height": 2.8,
            "empty_atom_dist": 1.9,
            "note": "surface-work-function",
        }
    ]
    assert result["work_function_results"] == [{"work_function": 4.2}]


def test_public_note_entry_points_document_note_parameter():
    checks = [
        ("abacusagent/modules/abacus.py", "abacus_prepare"),
        ("abacusagent/modules/scf.py", "abacus_calculation_scf"),
        ("abacusagent/modules/tool_wrapper.py", "prepare_abacus_inputs"),
        ("abacusagent/modules/tool_wrapper.py", "abacus_calculation_scf"),
        ("abacusagent/modules/submodules/abacus.py", "abacus_prepare"),
        ("abacusagent/modules/submodules/scf.py", "abacus_calculation_scf"),
        ("abacusagent/modules/submodules/band.py", "abacus_cal_band"),
        ("abacusagent/modules/submodules/jdos_pyatb.py", "pyatb_calculate_jdos"),
        ("abacusagent/modules/util/work_path.py", "generate_work_path"),
    ]
    offenders = []
    for relative_path, function_name in checks:
        path = TEST_ROOT / "src" / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        args = function.args.posonlyargs + function.args.args + function.args.kwonlyargs
        if not any(arg.arg == "note" for arg in args):
            offenders.append(f"src/{relative_path}:{function_name}:missing note parameter")
            continue
        docstring = ast.get_docstring(function) or ""
        if "note" not in docstring:
            offenders.append(f"src/{relative_path}:{function_name}:missing note doc")

    assert offenders == []
