"""Phono3py thermal closure shared by split and one-shot ABACUS routes.

The module deliberately keeps the runtime package contract explicit: the
selected ``abacus:1.1.0`` SIF supplies the version/API evidence, while this
adapter never invents a phono3py version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from abacusagent.modules.util.comm import run_abacus


def run_phono3py_thermal(
    *,
    work_dir: str | Path,
    displacement_yaml: str | Path,
    displacement_jobs: Sequence[str | Path] = (),
    mesh: Sequence[int] = (2, 2, 2),
    temperatures: Sequence[float] = (300.0,),
    cutoff_frequency: float | None = None,
    runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run force jobs, close FC2/FC3 and BTE, and return canonical artifacts."""
    from toolkits.phono3py_thermal import validate_runtime_contract, validate_thermal_options

    options = validate_thermal_options(mesh=mesh, temperatures=temperatures, cutoff_frequency=cutoff_frequency)
    contract = validate_runtime_contract(runtime_contract or {})
    root = Path(work_dir).resolve()
    yaml_path = Path(displacement_yaml).resolve()
    if not yaml_path.is_file() or yaml_path.parent != root:
        raise ValueError("displacement_yaml must be an existing canonical file in work_dir")
    jobs = [Path(item).resolve() for item in displacement_jobs]
    if jobs:
        for job in jobs:
            if not job.is_dir() or root not in job.parents:
                raise ValueError("displacement jobs must be existing directories contained in work_dir")
        run_abacus(jobs)
    import phono3py
    import dpdata
    import numpy as np

    ph3 = phono3py.load(str(yaml_path))
    if jobs:
        forces, energies = [], []
        for job in jobs:
            try:
                labelled = dpdata.LabeledSystem(str(job), fmt="abacus/scf")
                forces.append(labelled["forces"][0])
                energies.append(labelled["energies"][0])
            except Exception as exc:
                raise RuntimeError(f"failed to collect force/energy from displacement job {job}") from exc
        if not forces:
            raise RuntimeError("no displacement force/energy records were collected")
        ph3.forces = np.asarray(forces)
        ph3.supercell_energies = np.asarray(energies)
    if not hasattr(ph3, "produce_fc3") or not hasattr(ph3, "run_thermal_conductivity"):
        raise RuntimeError("installed phono3py API lacks FC3/BTE closure methods")
    ph3.produce_fc3()
    try:
        ph3.run_thermal_conductivity(mesh=options["mesh"], temperatures=options["temperatures"], cutoff_frequency=options["cutoff_frequency"])
    except TypeError:
        # Older compatible APIs omit cutoff_frequency when it is unset.
        if options["cutoff_frequency"] is not None:
            raise
        ph3.run_thermal_conductivity(mesh=options["mesh"], temperatures=options["temperatures"])
    kappa_candidates = sorted(root.glob("kappa*.hdf5")) + sorted(root.glob("kappa*.h5"))
    gamma_candidates = sorted(root.glob("gamma*.hdf5")) + sorted(root.glob("gamma*.h5"))
    if not kappa_candidates:
        raise RuntimeError("phono3py BTE completed without a canonical kappa HDF5 artifact")
    kappa_path = kappa_candidates[0]
    gamma_path = gamma_candidates[0] if gamma_candidates else None
    if hasattr(ph3, "save"):
        ph3.save(filename=str(root / "phono3py_thermal.yaml"))
    metadata = {"runtime": contract, "options": options, "displacement_yaml": yaml_path.name}
    (root / "phono3py_runtime_contract.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"work_dir": str(root), "kappa_hdf5": str(kappa_path), "gamma_hdf5": str(gamma_path) if gamma_path is not None else None, "runtime": contract, "options": options}
