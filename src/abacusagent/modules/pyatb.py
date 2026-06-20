import os
from pathlib import Path
from typing import Dict, Any, Literal

from abacustest.lib_prepare.abacus import ReadInput
from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.jdos_pyatb import pyatb_calculate_jdos as _pyatb_calculate_jdos

@mcp.tool()
def pyatb_calculate_jdos(
    abacus_inputs_dir: Path,
    note=None,
) -> Dict[str, Any]:
    """
    Plot the joint density of states (JDOS) using pyatb after ABACUS SCF calculation.
    Args:
        abacus_inputs_path (Path): The path to the ABACUS input files.
        note: Optional task label used to name generated work directories. Agent-facing wrappers must pass a non-empty note; None is kept for backward-compatible internal calls.
    Returns:
        Dict[str, Any]: A dictionary containing path to the plotted JDOS.
    """
    return _pyatb_calculate_jdos(abacus_inputs_dir, note=note)
