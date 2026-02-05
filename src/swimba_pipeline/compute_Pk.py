import argparse
import numpy as np
import typing as T
import h5py
from pathlib import Path
import Pk_library as PKL
import MAS_library as MASL

import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--target", type=Path, default=Path.cwd())
    parser.add_argument(
        "-p",
        "--parallel",
        help="How many parallel tasks to run.",
        type=int,
        default=joblib.cpu_count(),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--dm",
        action="store_true",
        help="Force snapshots to be considered as dark matter only",
    )

    args = parser.parse_args()
    snapshots: list[Path] = args.snapshots
    target: Path = args.target
    parallelism: int = args.parallel
    verbose: bool = args.verbose
    force_dm_only: bool = args.dm

    target.mkdir(parents=True, exist_ok=True)

    for snapshot in snapshots:
        with h5py.File(snapshot) as f:
            z = f["Header"].attrs["Redshift"]
            suffix = f"z={z:.2f}.txt"

            dm_only: bool = force_dm_only or ("PartType0" not in f)

            pk = compute_Pk(f, [0, 1, 4, 5], 512, "CIC", parallelism, verbose)
            np.savetxt(target / ("Pk_m_" + suffix), pk.T, delimiter="\t")

            if not dm_only:
                pk = compute_Pk(f, [0], 512, "CIC", parallelism, verbose)
                np.savetxt(target / ("Pk_g_" + suffix), pk.T, delimiter="\t")

                pk = compute_Pk(f, [1], 512, "CIC", parallelism, verbose)
                np.savetxt(target / ("Pk_c_" + suffix), pk.T, delimiter="\t")

                pk = compute_Pk(f, [4], 512, "CIC", parallelism, verbose)
                np.savetxt(target / ("Pk_s_" + suffix), pk.T, delimiter="\t")

                pk = compute_Pk(f, [5], 512, "CIC", parallelism, verbose)
                np.savetxt(target / ("Pk_bh_" + suffix), pk.T, delimiter="\t")


def compute_Pk(
    snapshot: h5py.File,
    ptypes: int | list[int],
    grid: int,
    MAS: T.Literal["CIC", "NGP", "TSC", "PCS"] = "CIC",
    threads: int = 1,
    verbose: bool = False,
):
    if isinstance(ptypes, int):
        ptypes = [ptypes]

    header = snapshot["Header"].attrs
    num_particles: int = np.sum(header["NumPart_Total"][:][ptypes])

    mass_table = header["MassTable"][:]
    box_size = header["BoxSize"] / 1e3  # Mpc / h

    coordinates = np.empty((num_particles, 3), dtype=np.float32)
    masses = np.empty(num_particles, dtype=np.float32)

    index = 0
    for ty in ptypes:
        name = f"PartType{ty}"
        if name in snapshot:
            group: h5py.Group = snapshot[name]

            coords_dset: h5py.Dataset = group["Coordinates"]
            num_in_group: int = coords_dset.shape[0]
            assert (
                index + num_in_group <= num_particles
            ), f"Malformed snapshot! {snapshot.filename}"
            coords_dset.read_direct(
                coordinates,
                np.s_[0:num_in_group, 0:3],
                np.s_[index : index + num_in_group, 0:3],
            )

            if "Masses" in group:
                masses_dset: h5py.Dataset = group["Masses"]
                masses_dset.read_direct(
                    masses,
                    np.s_[0:num_in_group],
                    np.s_[index : index + num_in_group],
                )
            else:
                # Check the mass table in the header
                masses[index : index + num_in_group] = mass_table[ty]

            index += num_in_group

    assert index == num_particles, f"Malformed snapshot! {snapshot.filename}"

    masses *= 1e10  # Msun / h
    coordinates /= 1e3  # Mpc / h

    density = np.zeros((grid,) * 3, dtype=np.float32)
    MASL.MA(coordinates, density, box_size, MAS, W=masses, verbose=verbose)
    density /= np.mean(density)
    density -= 1.0

    pk = PKL.Pk(density, box_size, 0, MAS, threads, verbose=verbose)
    return np.array([pk.k3D, pk.Pk[:, 0]])


if __name__ == "__main__":
    main()
