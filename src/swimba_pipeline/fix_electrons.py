import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import yaml

from .make_run import make_run, write_params, copy_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_name", type=Path)
    parser.add_argument("param_file", type=Path)
    parser.add_argument(
        "work_directory", nargs="?", type=Path, default=Path("./electron_fix")
    )
    parser.add_argument("-p", type=int, default=os.cpu_count() or 64)
    args = parser.parse_args()
    snapshot: Path = args.snapshot_name.resolve()
    param_file: Path = args.param_file.resolve()
    work_directory: Path = args.work_directory.resolve()
    parallelism: int = args.p

    with h5py.File(snapshot) as f:
        a = f["Header"].attrs["Scale-factor"][0]
        params = {}
        for k, v in f["Parameters"].attrs.items():
            group, name = k.split(":")
            if group not in params:
                params[group] = {}
            params[group][name] = v

    params["Cosmology"]["a_begin"] = a
    params["Cosmology"]["a_end"] = 1.1
    params["Snapshots"]["scale_factor_first"] = 1
    params["Statistics"]["scale_factor_first"] = 1
    params["FOF"]["scale_factor_first"] = 1
    params["TimeIntegration"]["dt_min"] = 1e-16

    work_directory.mkdir(parents=True, exist_ok=True)
    with open(work_directory / "params.yml", "w") as f:
        write_params(f, params)

    print("copying miscellaneous files...")
    copy_files(work_directory)

    shutil.copy(snapshot, work_directory / "ic.hdf5")

    with open(work_directory / "output.yml") as f:
        output = yaml.load(f, yaml.CLoader)["Default"]
    for k in output:
        output[k] = "off"
    output["ElectronNumberDensities_Gas"] = "DMantissa9"
    with open(work_directory / "output.yml", "w") as f:
        write_params(f, {"Default": output})

    os.chdir(work_directory)
    subprocess.run(
        [
            "/mnt/home/ssutherland/codes/swiftsim-e-fix/swift",
            "--pin",
            "--cosmology",
            "--simba",
            f"--threads={parallelism}",
            "params.yml",
        ],
        check=True,
    )

    with h5py.File("./snapshot_0000.hdf5") as f:
        data = f["PartType0/ElectronNumberDensities"][:]
    with h5py.File(snapshot, "a") as f:
        f["PartType0/ElectronNumberDensities"][:] = data

    # shutil.rmtree(work_directory)


if __name__ == "__main__":
    main()
