#!/bin/env python
import argparse
import ast
import shutil
import subprocess
import sys
import typing as T
import warnings
from io import TextIOBase
from pathlib import Path

import numpy as np
import yaml

from . import run_camb

Params = T.Dict[str, T.Dict[str, T.Any]]


def main():
    parser = argparse.ArgumentParser(
        prog="copy.py", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="directory in which new run should be created",
        default=Path.cwd(),
    )
    parser.add_argument(
        "variations",
        help="Change the arguments from the default.\n"
        "Each argument is of the form `Group:parameter=value`.\n"
        "Spaces don't strictly matter, but each argument as parsed by the shell translates to one variation.",
        nargs="*",
        type=Variation,
    )
    args = parser.parse_args()

    variations: Params = {}
    for v in args.variations:
        try:
            variations[v.group][v.name] = v.value
        except KeyError:
            variations[v.group] = {}
            variations[v.group][v.name] = v.value

    make_run(variations, args.target)


class Variation:
    group: str
    name: str
    value: T.Any

    def __init__(self, s: str):
        group, remainder = s.split(":")
        name, value = remainder.split("=")
        self.group = group.strip()
        self.name = name.strip()
        value = value.strip()
        try:
            self.value = ast.literal_eval(value)
        except ValueError as e:
            raise ValueError(f"Problem child: {s}") from e

    def __str__(self) -> str:
        return f"{self.group}:{self.name} = {self.value}"

    def __repr__(self) -> str:
        return str(self)


def make_run(
    variations: Params,
    target: Path,
    default_param_file: Path = Path(__file__).parent / "./data/default.yml",
):
    target.mkdir(parents=True, exist_ok=True)
    with open(default_param_file, "r") as handle:
        base_parameters: Params = yaml.load(handle, Loader=yaml.Loader)

    for group_name, group in variations.items():
        for name, value in group.items():
            try:
                base_parameters[group_name][name] = value
            except KeyError:
                base_parameters[group_name] = {}
                base_parameters[group_name][name] = value

    ic_params = base_parameters.pop("ICs", {})

    with open(target / "params.yml", "w") as f:
        write_params(f, base_parameters)

    print("cloning swiftsim...")
    clone_swift(target)
    print("initializing IC code...")
    setup_ICs(
        target,
        base_parameters["Cosmology"]["Omega_b"],
        base_parameters["Cosmology"]["Omega_cdm"],
        base_parameters["Cosmology"]["h"],
        ic_params["ns"],
        ic_params["sigma8"],
        ic_params["seed"],
    )
    print("copying miscellaneous files...")
    copy_files(target)


def write_params(stream: T.TextIO, parameters: Params):
    """
    Because yaml.safe_dump writes arrays like
    ```
    a:
    - 1
    - 2
    ```
    which isn't supported by the parser used in SWIFT, we need to create a writing function which
    writes arrays as
    ```
    a: [1, 2]
    ```
    """
    for group, params in sorted(parameters.items()):
        stream.write(group + ":\n")
        for name, value in sorted(params.items()):
            stream.write("  " + name + ": " + str(value) + "\n")


def clone_swift(target: Path):
    # We use the version copied into ceph so that the clones can be hardlinked to save disk space
    source = Path("/mnt/ceph/users/ssutherland/codes/swiftsim")
    assert source.exists(), "swiftsim code does not exist where expected"

    subprocess.run(
        [
            "git",
            "clone",
            str(source),
            str(target / "swiftsim"),
            "--branch=swimba-camels",
            "--quiet",
        ]
    )
    (target / "swift_mpi").symlink_to(Path("./swiftsim/swift_mpi"))


def setup_ICs(target: Path, Omega_b, Omega_cdm, h, ns, sigma_8, seed):
    source_2lpt = Path("/mnt/home/ssutherland/codes/2lpt/")
    assert source_2lpt.exists(), "2lpt code does not exist where expected"
    (target / "ICs").mkdir()
    shutil.copytree(
        source_2lpt, target / "ICs/2lpt", ignore=shutil.ignore_patterns(".git")
    )

    # copy the script used when generating the ICs
    source_convert: Path = Path(__file__).parent / "./convert_ic.py"
    assert source_convert.exists(), "IC conversion script does not exist where expected"

    # We do this rather than copying so we can change the shebang
    with open(source_convert, "r") as f:
        convert_ic_text = replace_shebang(f)

    with open(target / "ICs/convert_ic.py", "w") as f:
        f.write(convert_ic_text)
    (target / "ICs/convert_ic.py").chmod(0o755)

    power_spectrum, params = run_camb.run_camb(
        Omega_b=Omega_b, Omega_cdm=Omega_cdm, h=h, ns=ns
    )
    with open(target / "ICs/CAMB.params", "w") as f:
        f.write(str(params))
    np.savetxt(target / "ICs/Pk_m_z=0.000.txt", power_spectrum)

    lpt_param_file = f"""
Nmesh            512
Nsample          256
Box              25000.0
FileBase         ics
OutputDir        ./
GlassFile        ./2lpt/GLASS/dummy_glass_CDM_B_64_64.dat
GlassTileFac     4
Omega            {Omega_b + Omega_cdm}
OmegaLambda      {1 - Omega_b - Omega_cdm}
OmegaBaryon      0.049
OmegaDM_2ndSpecies  0.0
HubbleParam      0.6711
Redshift         127
Sigma8           {sigma_8}
SphereMode       0
WhichSpectrum    2
FileWithInputSpectrum   ./Pk_m_z=0.000.txt
InputSpectrum_UnitLength_in_cm  3.085678e24
ShapeGamma       0.201
PrimordialIndex  1.0

Phase_flip          0
RayleighSampling    1
Seed                {seed}

NumFilesWrittenInParallel 8
UnitLength_in_cm          3.085678e21
UnitMass_in_g             1.989e43
UnitVelocity_in_cm_per_s  1e5

WDM_On               0
WDM_Vtherm_On        0
WDM_PartMass_in_kev  10.0
"""
    with open(target / "ICs/2LPT.param", "w") as f:
        f.write(lpt_param_file)


def copy_files(target: Path):
    copy_files: Path = Path(__file__).parent / "data/copy_files"
    shutil.copytree(copy_files, target, dirs_exist_ok=True)


def replace_shebang(file: TextIOBase):
    shebang = "#!" + sys.executable + "\n"
    file_shebang = file.readline()
    if file_shebang.startswith("#!"):
        # Yup it's the shebang
        # Read the rest of the file
        # And prepend the current interpreter as the shebang
        result: str = shebang + file.read()
    else:
        # It wasn't a shebang?
        # Better include it just in case
        # No extra \n because readline() includes the \n already
        result: str = shebang + file_shebang + file.read()
        warnings.warn(f"file {file} did not start with a shebang")
    return result


if __name__ == "__main__":
    main()
