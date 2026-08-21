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
    displacement_jobs: Sequence[Any] = (),
    mesh: Sequence[int] = (2, 2, 2),
    temperatures: Sequence[float] = (300.0,),
    cutoff_frequency: float | None = None,
    isotope: bool | None = None,
    use_N_U: bool | None = None,
    boundary_mfp: float | None = None,
    boundary_length: float | None = None,
    runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run force jobs, close FC2/FC3 and BTE, and return canonical artifacts."""
    from toolkits.phono3py_thermal import validate_runtime_contract, validate_thermal_options

    options = validate_thermal_options(
        mesh=mesh, temperatures=temperatures, cutoff_frequency=cutoff_frequency,
        isotope=isotope, use_N_U=use_N_U, boundary_mfp=boundary_mfp,
        boundary_length=boundary_length,
    )
    # The expected version/API snapshot is a required input.  An empty
    # mapping must never make a live runtime self-validate against itself.
    contract = validate_runtime_contract(runtime_contract or {})
    root = Path(work_dir).resolve()
    yaml_path = Path(displacement_yaml).resolve()
    if not yaml_path.is_file() or yaml_path.parent != root:
        raise ValueError("displacement_yaml must be an existing canonical file in work_dir")
    typed_jobs = []
    for item in displacement_jobs:
        if isinstance(item, Mapping):
            typed_jobs.append((Path(item["path"]).resolve(), str(item.get("kind", "fc3")).lower()))
        else:
            typed_jobs.append((Path(item).resolve(), "fc3"))
    jobs = [path for path, _kind in typed_jobs]
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
        forces_fc3, energies_fc3, forces_fc2, energies_fc2 = [], [], [], []
        for job, kind in typed_jobs:
            try:
                labelled = dpdata.LabeledSystem(str(job), fmt="abacus/scf")
                if kind == "fc2":
                    forces_fc2.append(labelled["forces"][0])
                    energies_fc2.append(labelled["energies"][0])
                else:
                    forces_fc3.append(labelled["forces"][0])
                    energies_fc3.append(labelled["energies"][0])
            except Exception as exc:
                raise RuntimeError(f"failed to collect force/energy from displacement job {job}") from exc
        if not forces_fc3:
            raise RuntimeError("no displacement force/energy records were collected")
        expected_fc3 = len(ph3.dataset.get("displacements", ph3.dataset.get("first_atoms", [])))
        if expected_fc3 and expected_fc3 != len(forces_fc3):
            raise RuntimeError(f"FC3 displacement/force cardinality mismatch: {expected_fc3} != {len(forces_fc3)}")
        ph3.forces = np.asarray(forces_fc3)
        ph3.supercell_energies = np.asarray(energies_fc3)
        if forces_fc2:
            phonon_dataset = getattr(ph3, "phonon_dataset", {})
            expected_fc2 = len(phonon_dataset.get("displacements", phonon_dataset.get("first_atoms", []))) if isinstance(phonon_dataset, Mapping) else 0
            if expected_fc2 and expected_fc2 != len(forces_fc2):
                raise RuntimeError(f"FC2 displacement/force cardinality mismatch: {expected_fc2} != {len(forces_fc2)}")
            # Phono3py names the second-order force payload
            # ``phonon_forces`` (not ``forces_fc2``).
            ph3.phonon_forces = np.asarray(forces_fc2)
            ph3.phonon_supercell_energies = np.asarray(energies_fc2)
    if not hasattr(ph3, "produce_fc3") or not hasattr(ph3, "run_thermal_conductivity"):
        raise RuntimeError("installed phono3py API lacks FC3/BTE closure methods")
    ph3.produce_fc3()
    try:
        ph3.run_thermal_conductivity(
            mesh=options["mesh"], temperatures=options["temperatures"],
            cutoff_frequency=options["cutoff_frequency"], is_isotope=options["isotope"],
            is_N_U=options["use_N_U"], boundary_mfp=options["boundary_mfp"],
            write_kappa=True, write_gamma=True,
        )
    except TypeError:
        # Canonical phono3py 4.x consumes mesh/cutoff through the loaded
        # object and names the controls is_isotope/is_N_U.  Keep the first
        # call for older closure-compatible adapters, then retry using the
        # installed API's stable argument names when it rejects extras.
        ph3.run_thermal_conductivity(
            temperatures=options["temperatures"], is_isotope=options["isotope"],
            is_N_U=options["use_N_U"], boundary_mfp=options["boundary_mfp"],
            write_kappa=True, write_gamma=True,
        )
    kappa_candidates = sorted(root.glob("kappa*.hdf5")) + sorted(root.glob("kappa*.h5"))
    if not kappa_candidates:
        raise RuntimeError("phono3py BTE completed without a canonical kappa HDF5 artifact")
    gamma_candidates = []
    try:
        import h5py
        with h5py.File(kappa_candidates[0], "r") as handle:
            names = []
            handle.visit(names.append)
            gamma_candidates = [kappa_candidates[0]] if any(name.rsplit("/", 1)[-1].startswith("gamma") for name in names) else []
    except Exception:
        gamma_candidates = []
    kappa_path = kappa_candidates[0]
    gamma_path = gamma_candidates[0] if gamma_candidates else None
    if hasattr(ph3, "save"):
        ph3.save(filename=str(root / "phono3py_thermal.yaml"))
    metadata = {"runtime": contract, "options": options, "displacement_yaml": yaml_path.name}
    (root / "phono3py_runtime_contract.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"work_dir": str(root), "kappa_hdf5": str(kappa_path), "gamma_hdf5": str(gamma_path) if gamma_path is not None else None, "runtime": contract, "options": options}
