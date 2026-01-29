import argparse
import sys
import typing as T
from collections import OrderedDict
from pathlib import Path

import h5py
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "parameter_file_type",
        help="What kind of parameter file to convert into",
        choices=["subfind"],
    )
    parser.add_argument(
        "source_params", type=Path, help="SWIFT parameter file to be converted"
    )
    parser.add_argument(
        "destination_params", type=Path, help="Path to new converted parameter file"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("additional_changes", nargs="*")

    args = parser.parse_args()

    destination_type: str = args.parameter_file_type
    source: Path = args.source_params
    dest: Path = args.destination_params
    verbose: bool = args.verbose
    force: bool = args.force
    extra_changes = args.additional_changes

    if not source.exists():
        print(f"File `{source}` does not exist!", file=sys.stderr)

    if dest.exists():
        if force:
            dest.unlink()
        else:
            print(
                f"File `{dest}` already exists. Use --force to remove it.",
                file=sys.stderr,
            )

    if verbose:
        print(f"reading `{source}`")
    with open(source) as f:
        source_params = yaml.load(f, yaml.CLoader)

    if destination_type == "subfind":
        convert_to_subfind(source_params, dest, verbose, extra_changes)
    else:
        print(f"Unrecognized file type: `{destination_type}`", file=sys.stderr)
        print("Allowed file types are: `subfind`", file=sys.stderr)


def convert_to_subfind(
    source_params: T.Any, dest: Path, verbose: bool, extra_changes: list[str]
):
    default_params_file = Path(__file__).parent / "./data/subfind_params.txt"

    params: OrderedDict[str, str] = OrderedDict()

    with open(default_params_file) as f:
        while line := f.readline():
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            name, value = parts
            params[name] = value.strip()

    h = source_params["Cosmology"]["h"]
    params["Omega0"] = (
        source_params["Cosmology"]["Omega_cdm"] + source_params["Cosmology"]["Omega_b"]
    )
    params["OmegaLambda"] = source_params["Cosmology"]["Omega_lambda"]
    params["OmegaBaryon"] = source_params["Cosmology"]["Omega_b"]
    params["HubbleParam"] = h
    params["BoxSize"] = get_boxsize(source_params) * h * 1000
    params["SofteningMaxPhysType1"] = source_params["Gravity"][
        "max_physical_DM_softening"
    ]
    params["SofteningComovingType1"] = source_params["Gravity"]["comoving_DM_softening"]
    for p in (0, 2, 3, 4):
        params[f"SofteningMaxPhysType{p}"] = source_params["Gravity"][
            "max_physical_baryon_softening"
        ]
        params[f"SofteningComovingType{p}"] = source_params["Gravity"][
            "comoving_baryon_softening"
        ]

    for c in extra_changes:
        name, value = c.split("=")
        params[name] = value

    with open(dest, "w") as f:
        if verbose:
            print(f"Writing to `{dest}`")
        for key, value in params.items():
            f.write(f"{key: <50}{value}\n")


def get_boxsize(source_params: T.Any) -> float:
    ic_file = source_params["InitialConditions"]["file_name"]
    with h5py.File(ic_file) as f:
        return f["Header"].attrs["BoxSize"].mean()


if __name__ == "__main__":
    main()
