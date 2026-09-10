"""Regression tests for ABACUS MD trajectory metadata."""

from pathlib import Path

import numpy as np
from ase.io import read

from abacusagent.modules.submodules.md import convert_md_dump_to_ase_traj


MD_DUMP = """MDSTEP: 0
LATTICE_CONSTANT: 3.335091
LATTICE_VECTORS
1 0 0
0 1 0
0 0 1
INDEX    LABEL
0 Mg 0.000000 0.000000 0.000000 0 0 0 0 0 0
1 O 1.667545 1.667545 1.667545 0 0 0 0 0 0

MDSTEP: 1
LATTICE_CONSTANT: 3.335091
LATTICE_VECTORS
1 0 0
0 1 0
0 0 1
INDEX    LABEL
0 Mg 0.100000 0.000000 0.000000 0 0 0 0 0 0
1 O 1.667545 1.667545 1.667545 0 0 0 0 0 0

"""


def test_md_dump_conversion_preserves_periodic_cell_metadata(tmp_path: Path, monkeypatch):
    """Catch conversion dropping PBC flags from every generated MD frame."""
    md_dump = tmp_path / "MD_dump"
    md_dump.write_text(MD_DUMP, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    trajectory, frame_count = convert_md_dump_to_ase_traj(md_dump)

    frames = read(str(trajectory), index=":")
    assert frame_count == 2
    assert len(frames) == 2
    expected_cell = np.eye(3) * 3.335091
    for frame in frames:
        assert np.allclose(frame.cell.array, expected_cell)
        assert np.array_equal(frame.get_pbc(), [True, True, True])
