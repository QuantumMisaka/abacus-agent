import os

from abacusagent.modules.util import comm


def test_structured_abacus_runtime_env_builds_mpi_command(monkeypatch):
    monkeypatch.setenv("ABACUS_COMMAND", "mpirun -np 8 --map-by ppr:1:node abacus")
    monkeypatch.setenv("ABACUS_OMP_THREADS", "2")
    monkeypatch.setenv("ABACUS_MPI_PROCS", "4")
    monkeypatch.setenv("ABACUS_MPI_LAUNCHER", "ignored")

    command, env = comm.resolve_abacus_runtime_command(physical_cores=16)

    assert command == "mpirun -np 8 --map-by ppr:1:node abacus"
    assert env["OMP_NUM_THREADS"] == "2"


def test_structured_abacus_runtime_env_builds_openmp_command(monkeypatch):
    monkeypatch.setenv("ABACUS_COMMAND", "abacus")
    monkeypatch.setenv("ABACUS_OMP_THREADS", "6")

    command, env = comm.resolve_abacus_runtime_command(physical_cores=16)

    assert command == "abacus"
    assert env["OMP_NUM_THREADS"] == "6"


def test_abacus_command_takes_precedence_over_structured_mpi_metadata(monkeypatch):
    monkeypatch.setenv("ABACUS_COMMAND", "OMP_NUM_THREADS=1 mpirun -np 16 abacus")
    monkeypatch.setenv("ABACUS_MPI_PROCS", "4")
    monkeypatch.setenv("ABACUS_OMP_THREADS", "3")

    command, env = comm.resolve_abacus_runtime_command(physical_cores=16)

    assert command == "OMP_NUM_THREADS=1 mpirun -np 16 abacus"
    assert env["OMP_NUM_THREADS"] == "3"


def test_legacy_abacus_command_remains_supported(monkeypatch):
    legacy = "OMP_NUM_THREADS=1 mpirun -np 16 abacus"
    monkeypatch.setenv("ABACUS_COMMAND", legacy)
    monkeypatch.delenv("ABACUS_MPI_PROCS", raising=False)
    monkeypatch.delenv("ABACUS_OMP_THREADS", raising=False)
    monkeypatch.delenv("ABACUS_EXECUTABLE", raising=False)
    monkeypatch.delenv("ABACUS_MPI_LAUNCHER", raising=False)

    command, env = comm.resolve_abacus_runtime_command(physical_cores=16)

    assert command == legacy
    assert env is None


def test_run_abacus_passes_structured_env_to_run_command(tmp_path, monkeypatch):
    calls = []

    def fake_run_command(cmd, shell=True, env=None):
        calls.append((cmd, shell, env))
        return 0, "", ""

    monkeypatch.setenv("ABACUSAGENT_SUBMIT_TYPE", "local")
    monkeypatch.setenv("ABACUS_COMMAND", "abacus")
    monkeypatch.setenv("ABACUS_OMP_THREADS", "5")
    monkeypatch.setattr(comm, "get_physical_cores", lambda: 16)
    monkeypatch.setattr(comm, "run_command", fake_run_command)

    comm.run_abacus(tmp_path, log_file="abacus.log")

    assert calls[0][0] == ["abacus > abacus.log 2>&1"]
    assert calls[0][2]["OMP_NUM_THREADS"] == "5"
    assert os.getcwd() != str(tmp_path)


def test_run_pyatb_passes_omp_threads_to_run_command(tmp_path, monkeypatch):
    calls = []

    def fake_run_command(cmd, shell=True, env=None):
        calls.append((cmd, shell, env))
        return 0, "", ""

    monkeypatch.setenv("PYATB_COMMAND", "mpirun -np 8 --map-by ppr:1:node pyatb")
    monkeypatch.setenv("PYATB_OMP_THREADS", "1")
    monkeypatch.setattr(comm, "run_command", fake_run_command)

    comm.run_pyatb(tmp_path)

    assert calls[0][0] == "mpirun -np 8 --map-by ppr:1:node pyatb"
    assert calls[0][2]["OMP_NUM_THREADS"] == "1"
    assert os.getcwd() != str(tmp_path)


def test_run_pyatb_legacy_command_remains_supported(tmp_path, monkeypatch):
    calls = []

    def fake_run_command(cmd, shell=True, env=None):
        calls.append((cmd, shell, env))
        return 0, "", ""

    monkeypatch.setenv("PYATB_COMMAND", "OMP_NUM_THREADS=1 pyatb")
    monkeypatch.delenv("PYATB_OMP_THREADS", raising=False)
    monkeypatch.setattr(comm, "run_command", fake_run_command)

    comm.run_pyatb(tmp_path)

    assert calls[0] == ("OMP_NUM_THREADS=1 pyatb", True, None)
