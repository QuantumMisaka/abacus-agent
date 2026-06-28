from pathlib import Path
from typing import List, Dict, Any, Literal, Optional

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.work_function import abacus_cal_work_function as _abacus_cal_work_function

@mcp.tool()
def abacus_cal_work_function(
    abacus_inputs_dir: Path,
    vacuum_direction: Literal['a', 'b', 'c', 'auto'] = 'c',
    dipole_correction: bool = False,
    work_function_threshold: float = 0.01,
    use_empty_atom: bool = False,
    empty_atom_elem: Optional[str] = None,
    empty_atom_height: float = 2.0,
    empty_atom_dist: float = 2.0,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate the electrostatic potential and work function using ABACUS.

    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        vacuum_direction (Literal['a', 'b', 'c', 'auto']): The direction of the vacuum. If set to auto, the direction will try to be determined automatically.
        dipole_correction (bool): Whether to apply dipole correction along the vacuum direction. For polar slabs, it is recommended to enable dipole correction.
        work_function_threshold (float): Plateau detection threshold used by abacustest work-function post-processing.
        use_empty_atom (bool): Whether to add empty atoms in the vacuum region before running the work-function calculation.
        empty_atom_elem (str or None): Element symbol used for empty atoms. If None, abacustest uses the first element in the structure.
        empty_atom_height (float): Distance from surface edge to the empty atom layer.
        empty_atom_dist (float): Approximate in-plane spacing between empty atoms.
        note: Optional task label used to name generated work directories. Agent-facing wrappers should pass a non-empty note; None is kept for backward-compatible internal calls.

    Returns:
        A dictionary containing:
        - elecstat_pot_work_function_work_path (Path): Path to the ABACUS job directory calculating electrostatic potential and work function.
        - elecstat_pot_file (Path): Path to the cube file containing the electrostatic potential.
        - averaged_elecstat_pot_plot (Path): Path to the plot of the averaged electrostatic potential.
        - averaged_elecstat_pot_dat_file (Path): Path to the data used in the plot of averaged electrostatic potential.
        - work_function_results (list): A list of 1 or 2 dictionary. If dipole correction is not used, only 1 dictionaray will be returned.
          If dipole correction is used, there will be 2 dictionarys for calculated work function of 2 surfaces of the slab. Each dictionary contains 3 keys:
            - 'work_function': calculated work function
            - 'plateau_start_fractional': Fractional coordinate of start of the identified plateau in the given vacuum direction
            - 'plateau_end_fractional': Fractional coordinate of end of the identified plateau in the given vacuum direction
    """
    return _abacus_cal_work_function(
        abacus_inputs_dir=abacus_inputs_dir,
        vacuum_direction=vacuum_direction,
        dipole_correction=dipole_correction,
        work_function_threshold=work_function_threshold,
        use_empty_atom=use_empty_atom,
        empty_atom_elem=empty_atom_elem,
        empty_atom_height=empty_atom_height,
        empty_atom_dist=empty_atom_dist,
        note=note,
    )
