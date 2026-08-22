"""Phono3py thermal closure shared by split and one-shot ABACUS routes.

The module deliberately keeps the runtime package contract explicit: the
selected ``abacus:1.1.0`` SIF supplies the version/API evidence, while this
adapter never invents a phono3py version.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from abacusagent.modules.util.comm import run_abacus


def _dataset_count(dataset: Mapping[str, Any] | None) -> int:
    if not isinstance(dataset, Mapping):
        return 0
    displacements = dataset.get("displacements")
    if displacements is not None:
        return len(displacements)
    first_atoms = dataset.get("first_atoms")
    if first_atoms is not None:
        # Type-I FC3 datasets nest pair displacements under each first atom's
        # ``second_atoms``; every first- and second-atom entry maps to one
        # displacement job, so the flat job count is the flattened sum.
        return sum(1 + len(entry.get("second_atoms") or []) for entry in first_atoms)
    return 0


def _is_type_ii_dataset(dataset: Any) -> bool:
    """Type-II (random snapshot) datasets carry a flat ``displacements`` key.

    Real phono3py dataset mappings have no ``type`` key; the displacement
    layout is the only reliable discriminator across supported versions.
    """
    return isinstance(dataset, Mapping) and dataset.get("displacements") is not None


def _require_symfc() -> None:
    try:
        import symfc  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "random (Type-II) phono3py displacements require the symfc "
            "force-constants calculator, which is not installed in this runtime"
        ) from exc


_SCF_CONVERGED_TOKEN = "charge density convergence is achieved"


def _scf_log_converged(path: Path) -> bool:
    """Return True only when the SCF log proves density convergence.

    ABACUS writes ``abacus.json`` on *normal termination* — including an
    scf_nmax exhaustion exit — so a marker alone must never count as
    converged.  A missing or unreadable log fails closed (job recomputed):
    recomputing one displacement SCF is cheap, collecting unconverged forces
    silently corrupts FC2/FC3.
    """
    log = path / "OUT.ABACUS" / "running_scf.log"
    try:
        return _SCF_CONVERGED_TOKEN in log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _job_is_completed(path: Path) -> bool:
    """Return True when a displacement job already has a *converged*, collectable result.

    Checkpoint-resume treats a job as finished only when its SCF log proves
    charge-density convergence (see :func:`_scf_log_converged`) AND the
    directory carries a collectable result: a readable ``abacus.json``
    completion marker (written by ABACUS on normal termination) or a single
    SCF frame loadable by dpdata.  The latter also recognizes jobs computed
    by earlier one-shot runs that only left ``OUT.ABACUS`` behind.  Both
    signals are only heuristics for skipping SCF compute; the final
    force/energy collection still enforces the full-displacement cardinality
    contract.
    """
    if not _scf_log_converged(path):
        return False
    marker = path / "abacus.json"
    if marker.is_file():
        try:
            json.loads(marker.read_text(encoding="utf-8"))
            return True
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    try:
        import dpdata

        labelled = dpdata.LabeledSystem(str(path), fmt="abacus/scf")
        forces = labelled["forces"]
        energies = labelled["energies"]
        return len(forces) == 1 and len(energies) == 1
    except Exception:
        return False


def collect_force_energy_records(
    ph3: Any,
    jobs: Sequence[tuple[Path, str]],
    *,
    work_dir: Path,
) -> dict[str, Any]:
    """Collect displacement records with one strict, shared contract.

    Both split collection and one-shot closure use this helper.  A missing,
    unreadable, or out-of-order job is a hard failure: silently dropping a
    force/energy row changes the displacement-to-force mapping and can make a
    formally successful BTE physically meaningless.
    """
    import dpdata
    import numpy as np

    root = work_dir.resolve()
    grouped: dict[str, list[tuple[Path, Any, Any]]] = {"fc3": [], "fc2": []}
    for raw_path, raw_kind in jobs:
        path = Path(raw_path).resolve()
        kind = str(raw_kind or "fc3").lower()
        if kind not in grouped:
            raise ValueError(f"unsupported phono3py displacement kind: {kind}")
        if not path.is_dir() or root not in path.parents:
            raise ValueError("displacement jobs must be existing directories contained in work_dir")
        try:
            labelled = dpdata.LabeledSystem(str(path), fmt="abacus/scf")
            forces = labelled["forces"]
            energies = labelled["energies"]
            if len(forces) != 1 or len(energies) != 1:
                raise ValueError("each ABACUS displacement job must contain exactly one frame")
            grouped[kind].append((path, forces[0], energies[0]))
        except Exception as exc:
            raise RuntimeError(f"failed to collect force/energy from displacement job {path}") from exc

    expected_fc3 = _dataset_count(getattr(ph3, "dataset", None))
    expected_fc2 = _dataset_count(getattr(ph3, "phonon_dataset", None))
    for kind, expected in (("fc3", expected_fc3), ("fc2", expected_fc2)):
        actual = len(grouped[kind])
        if expected and actual != expected:
            raise RuntimeError(f"{kind.upper()} displacement/force cardinality mismatch: {expected} != {actual}")
    if not grouped["fc3"]:
        raise RuntimeError("no FC3 displacement force/energy records were collected")

    return {
        "forces_fc3": np.asarray([row[1] for row in grouped["fc3"]]),
        "energies_fc3": np.asarray([row[2] for row in grouped["fc3"]]),
        "forces_fc2": np.asarray([row[1] for row in grouped["fc2"]]),
        "energies_fc2": np.asarray([row[2] for row in grouped["fc2"]]),
        "completed_jobs": [row[0].name for kind in ("fc3", "fc2") for row in grouped[kind]],
    }


def _set_compatibility_controls(ph3: Any, options: Mapping[str, Any]) -> None:
    """Configure controls through phono3py's actual object contract."""
    mesh = options["mesh"]
    if not hasattr(type(ph3), "mesh_numbers"):
        raise RuntimeError("phono3py object lacks the mesh_numbers property")
    ph3.mesh_numbers = mesh
    if list(ph3.mesh_numbers) != list(mesh):
        raise RuntimeError("phono3py mesh_numbers was not retained")
    cutoff = options["cutoff_frequency"]
    if cutoff is not None:
        # phono3py 4.x has no public setter; this is the constructor-owned
        # state used by _set_mesh_numbers and is deliberately verified.
        if not hasattr(ph3, "_cutoff_frequency"):
            raise RuntimeError("phono3py object lacks supported cutoff_frequency state")
        ph3._cutoff_frequency = float(cutoff)
        if float(ph3._cutoff_frequency) != float(cutoff):
            raise RuntimeError("phono3py cutoff_frequency was not retained")


def _init_phph_interaction_if_required(ph3: Any) -> None:
    initializer = getattr(ph3, "init_phph_interaction", None)
    if initializer is None:
        raise RuntimeError("installed phono3py API lacks init_phph_interaction")
    initializer()


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
    import phono3py

    # Load before running any job so a runtime that cannot close the loaded
    # dataset (e.g. Type-II without symfc) fails before burning SCF compute.
    ph3 = phono3py.load(str(yaml_path))
    fc3_calculator = None
    if _is_type_ii_dataset(getattr(ph3, "dataset", None)):
        _require_symfc()
        fc3_calculator = "symfc"
    fc2_calculator = None
    if _is_type_ii_dataset(getattr(ph3, "phonon_dataset", None)):
        _require_symfc()
        fc2_calculator = "symfc"
    if jobs:
        pending_jobs = [job for job in jobs if not _job_is_completed(job)]
        if pending_jobs:
            run_abacus(pending_jobs)
        records = collect_force_energy_records(ph3, typed_jobs, work_dir=root)
        ph3.forces = records["forces_fc3"]
        ph3.supercell_energies = records["energies_fc3"]
        if len(records["forces_fc2"]):
            # Phono3py names the second-order force payload
            # ``phonon_forces`` (not ``forces_fc2``).
            ph3.phonon_forces = records["forces_fc2"]
            ph3.phonon_supercell_energies = records["energies_fc2"]
    if not hasattr(ph3, "produce_fc3") or not hasattr(ph3, "run_thermal_conductivity"):
        raise RuntimeError("installed phono3py API lacks FC3/BTE closure methods")
    if fc3_calculator is not None:
        ph3.produce_fc3(fc_calculator=fc3_calculator)
    else:
        ph3.produce_fc3()
    # ``supercell_fc2`` is serialized by phono3py as the independent
    # ``phonon_supercell_matrix``.  In phono3py 4.1.0 this path does not
    # close FC2 as a side effect of ``produce_fc3``; it must be produced
    # before initializing the ph-ph interaction used by the BTE.
    if getattr(ph3, "phonon_supercell_matrix", None) is not None:
        if not hasattr(ph3, "produce_fc2"):
            raise RuntimeError("installed phono3py API lacks produce_fc2 for independent FC2 supercell")
        if fc2_calculator is not None:
            ph3.produce_fc2(fc_calculator=fc2_calculator)
        else:
            ph3.produce_fc2()
    # Symmetrize the force constants before the BTE step.  Unsymmetrized
    # FC3/FC2 from finite differences carry numerical noise that inflates
    # scattering and collapses kappa (systematic-111 benchmark evidence:
    # 5.57 W/mK unsymmetrized vs 122.46 W/mK with symmetrize_fc3+fc2).
    # Fail closed on runtimes lacking the API — never run an unsymmetrized
    # closure silently.
    if not hasattr(ph3, "symmetrize_fc3"):
        raise RuntimeError("installed phono3py API lacks symmetrize_fc3 for BTE closure")
    ph3.symmetrize_fc3()
    if not hasattr(ph3, "symmetrize_fc2"):
        raise RuntimeError("installed phono3py API lacks symmetrize_fc2 for BTE closure")
    ph3.symmetrize_fc2()
    # Configure the object as well as the call signature.  This is required
    # even for releases that still accept legacy keywords, because mesh and
    # cutoff are consumed by the interaction initialization path.
    _set_compatibility_controls(ph3, options)
    _init_phph_interaction_if_required(ph3)
    old_cwd = Path.cwd()
    try:
        os.chdir(root)
        try:
            ph3.run_thermal_conductivity(
                mesh=options["mesh"], temperatures=options["temperatures"],
                cutoff_frequency=options["cutoff_frequency"], is_isotope=options["isotope"],
                is_N_U=options["use_N_U"], boundary_mfp=options["boundary_mfp_api_um"],
                write_kappa=True, write_gamma=True,
            )
        except TypeError:
            # Some supported releases consume mesh/cutoff through the loaded
            # object.  Set and verify every control before retrying; a retry
            # that drops a requested control is scientifically unsafe.
            _set_compatibility_controls(ph3, options)
            _init_phph_interaction_if_required(ph3)
            ph3.run_thermal_conductivity(
                temperatures=options["temperatures"],
                is_isotope=options["isotope"], is_N_U=options["use_N_U"],
                boundary_mfp=options["boundary_mfp_api_um"],
                write_kappa=True, write_gamma=True,
            )
    finally:
        os.chdir(old_cwd)
    all_kappa = sorted(root.glob("kappa*.hdf5")) + sorted(root.glob("kappa*.h5"))
    # ``write_gamma=True`` also emits per-grid-point ``kappa-m<mesh>-g<gp>.hdf5``
    # files that sort before the canonical ``kappa-m<mesh>.hdf5``.  The canonical
    # thermal-conductivity tensor must not be confused with a gamma component.
    canonical = [path for path in all_kappa if not re.search(r"-g\d+\.(?:hdf5|h5)$", path.name)]
    kappa_candidates = canonical or all_kappa
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
