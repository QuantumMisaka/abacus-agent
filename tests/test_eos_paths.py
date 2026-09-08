"""EOS staging contracts for non-canonical STRU references."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput


EOS_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "abacusagent"
    / "modules"
    / "submodules"
    / "eos.py"
)


def _load_eos_module():
    spec = importlib.util.spec_from_file_location(
        "checked_out_abacusagent_modules_submodules_eos_paths", EOS_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(path: Path, stru_reference: str | None) -> Path:
    path.mkdir(parents=True)
    input_lines = [
        "INPUT_PARAMETERS",
        "calculation relax",
        "basis_type lcao",
        "pseudo_dir ./pp",
        "orbital_dir ./orb",
    ]
    if stru_reference is not None:
        input_lines.append(f"stru_file {stru_reference}")
    (path / "INPUT").write_text("\n".join(input_lines) + "\n", encoding="utf-8")
    (path / "STRU").write_text(
        """ATOMIC_SPECIES
H 1.008 H.upf

NUMERICAL_ORBITAL
H.orb

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
10 0 0
0 10 0
0 0 10

ATOMIC_POSITIONS
Direct
H
0.0
1
0.1 0.1 0.1
""",
        encoding="utf-8",
    )
    (path / "H.upf").write_text("pp\n", encoding="utf-8")
    (path / "H.orb").write_text("orb\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "reference_kind",
    ["default", "relative", "absolute"],
)
def test_eos_stages_each_stru_reference_as_private_point_inputs(
    monkeypatch, tmp_path, reference_kind
):
    source = _write_inputs(tmp_path / "inputs", None)
    if reference_kind == "relative":
        source.joinpath("INPUT").write_text(
            source.joinpath("INPUT").read_text(encoding="utf-8")
            + "stru_file ./STRU\n",
            encoding="utf-8",
        )
    elif reference_kind == "absolute":
        source.joinpath("INPUT").write_text(
            source.joinpath("INPUT").read_text(encoding="utf-8")
            + f"stru_file {source.joinpath('STRU').absolute()}\n",
            encoding="utf-8",
        )

    eos = _load_eos_module()
    work_path = tmp_path / f"eos-{reference_kind}"
    monkeypatch.setattr(eos, "check_abacus_inputs", lambda _path: (True, ""))
    monkeypatch.setattr(eos, "generate_work_path", lambda **_kwargs: str(work_path))
    monkeypatch.setattr(eos, "run_abacus", lambda _paths: None)
    monkeypatch.setattr(
        eos,
        "collect_metrics",
        lambda path: {
            "normal_end": True,
            "converge": True,
            "energy": -10.0 + int(Path(path).name.rsplit("_", 1)[1]),
        },
    )
    monkeypatch.setattr(
        eos,
        "eos_fit",
        lambda volumes, energies: (
            float(volumes[len(volumes) // 2]),
            float(energies[len(energies) // 2]),
            np.asarray(volumes),
            np.asarray(energies),
            100.0,
            4.0,
            0.0,
        ),
    )
    monkeypatch.setattr(
        eos,
        "plot_eos",
        lambda *_args: work_path.joinpath("fit.png"),
    )

    source_bytes = {
        name: source.joinpath(name).read_bytes() for name in ("INPUT", "STRU")
    }
    result = eos.abacus_eos(
        source,
        stru_scale_number=2,
        scale_stepsize=0.01,
        note=f"path-{reference_kind}",
    )

    assert "E0" in result
    points = sorted(work_path.glob("scale_cell_*"))
    assert len(points) == 5
    assert all(not point.joinpath("STRU").is_symlink() for point in points)
    original_length = np.linalg.norm(
        np.asarray(AbacusStru.ReadStru(source / "STRU").get_cell())[0]
    )
    lengths = [
        np.linalg.norm(np.asarray(AbacusStru.ReadStru(point / "STRU").get_cell())[0])
        for point in points
    ]
    assert lengths == pytest.approx(
        [original_length * scale for scale in (0.98, 0.99, 1.0, 1.01, 1.02)]
    )
    assert {
        name: source.joinpath(name).read_bytes() for name in ("INPUT", "STRU")
    } == source_bytes
    assert ReadInput(work_path / "input_stru" / "INPUT")["stru_file"] == "STRU"
