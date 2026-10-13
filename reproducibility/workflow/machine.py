"""Describe the machine and the environment where the experiments run.

``describe`` describes the processor, memory, operating system, Python, the
threading libraries and the settings that change the number of threads, which
each job stores with its results, since the jobs of a run may run on different
machines. Run as a script, it writes the exact packages of the environment.
"""

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
from threadpoolctl import threadpool_info


def run(command):
    """Return the output of a command, or ``None`` if it cannot run."""
    try:
        return subprocess.run(
            command, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def processor_model():
    """Return the model of the processor, which psutil does not give."""
    if sys.platform == "darwin":
        return run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        with key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or None


def describe():
    """Describe the machine, its threading libraries and their settings."""
    # threadpoolctl only finds the threading libraries that are loaded, so the
    # modules that load BLAS and OpenMP are imported first
    for module in ["numpy", "scipy.linalg", "sklr.neighbors"]:
        importlib.import_module(module)
    return {
        "processor": processor_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "memory_bytes": psutil.virtual_memory().total,
        "system": platform.platform(),
        "python": sys.version.split()[0],
        "threadpools": [
            {key: pool[key] for key in ["internal_api", "version", "num_threads"]}
            for pool in threadpool_info()
        ],
        "environment_variables": {
            name: os.environ.get(name)
            for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        },
    }


def export_environment():
    """List the exact packages of the environment.

    The package manager of the environment lists the packages that it installed,
    and pip those installed with pip, such as Snakemake and scikit-lr.
    """
    sections = []
    for name in ["micromamba", "mamba", "conda"]:
        executable = shutil.which(name)
        output = executable and run(
            [executable, "list", "--explicit", "--prefix", sys.prefix]
        )
        if output:
            lines = [line for line in output.splitlines() if line.strip()]
            sections.append("\n".join(lines))
            break
    sections.append(run([sys.executable, "-m", "pip", "freeze"]) or "")
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(export_environment() + "\n")


if __name__ == "__main__":
    main()
