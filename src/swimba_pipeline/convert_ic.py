#!/bin/env python
import ast
import math
import sys

import h5py
import numpy as np
import pygadgetreader


def main():
    if len(sys.argv) != 3:
        print(f"USAGE: {sys.argv[0]} [source file] [destination file]")
        return

    source = sys.argv[1]
    destination = sys.argv[2]

    copy_ic(source, destination)


def copy_ic(source: str, destination: str):
    with h5py.File(destination, "w") as f:
        header = {}
        original_header = pygadgetreader.readheader(source, "header")
        h = original_header["h"]
        a = original_header["time"]
        header["BoxSize"] = [original_header["boxsize"] / h / 1000] * 3
        header["Dimension"] = [3]
        header["Flag_Entropy_ICs"] = [0]
        header["MassTable"] = [0.0] * 7
        header["NumFilesPerSnapshot"] = [1]
        header["NumPart_ThisFile"] = [
            original_header["ngas"],
            original_header["ndm"],
        ] + [0] * 5
        header["NumPart_Total"] = [
            original_header["ngas"] & 0xFFFFFFFF,
            original_header["ndm"] & 0xFFFFFFFF,
        ] + [0] * 5
        header["NumPart_Total_HighWord"] = [
            original_header["ngas"] >> 32,
            original_header["ndm"] >> 32,
        ] + [0] * 5
        header["Redshift"] = [original_header["redshift"]]
        header["Time"] = [original_header["time"]]

        f_header = f.create_group("Header")
        f_header.attrs.update(header)

        units = {
            "Unit length in cgs (U_L)": [3.08567758149e24],  # Mpc in cm
            "Unit mass in cgs (U_M)": [1.98841e43],  # 1e10 Msun in g
            "Unit time in cgs (U_t)": [
                3.08567758149e24 / 1e5
            ],  # Mpc / (km / s) in cm/s
            "Unit current in cgs (U_I)": [1.0],  # A in A
            "Unit temperature in cgs (U_T)": [1.0],  # K in K
        }
        f_units = f.create_group("Units")
        f_units.attrs.update(units)

        length = 1 / h / 1000
        mass = 1 / h
        velocity = math.sqrt(a)

        fields = [
            ("pos", "gas", "PartType0/Coordinates", length),
            ("vel", "gas", "PartType0/Velocities", velocity),
            ("mass", "gas", "PartType0/Masses", mass),
            ("pid", "gas", "PartType0/ParticleIDs", 1),
            ("pos", "dm", "PartType1/Coordinates", length),
            ("vel", "dm", "PartType1/Velocities", velocity),
            ("mass", "dm", "PartType1/Masses", mass),
            ("pid", "dm", "PartType1/ParticleIDs", 1),
        ]

        for gadget_name, gadget_type, swift_name, conversion in fields:
            data = pygadgetreader.readsnap(source, gadget_name, gadget_type)
            print("Copying", swift_name)
            data *= conversion
            if gadget_name == "pos":
                # Make sure coordinates are withing the size of the box
                # Wrap them around if they're not.
                data %= header["BoxSize"]
            f[swift_name] = data

        # Set smoothing lengths to mean inter-particle separation
        hsm = header["BoxSize"][0] / np.cbrt(original_header["ngas"])
        f["PartType0/SmoothingLength"] = hsm * np.ones(original_header["ngas"])

        # Write internal energies
        # based on code from monofonic
        gamma = 5 / 3
        YHe = 0.2457519853817943
        M_gas = f["PartType0/Masses"][:].sum()
        M_dm = f["PartType1/Masses"][:].sum()
        Omega_b = M_gas / (M_gas + M_dm) * original_header["O0"]
        Tcmb0 = 2.7255

        npol = 1 / (gamma - 1) if math.fabs(1.0 - gamma) > 1e-7 else 1
        unitv = 1e5
        adec = 1.0 / (160.0 * (Omega_b * h * h / 0.022) ** (2.0 / 5.0))
        Tini = Tcmb0 / a if a < adec else Tcmb0 / a / a * adec
        mu = 4.0 / (8.0 - 5.0 * YHe) if Tini > 1e4 else 4.0 / (1.0 + 3.0 * (1.0 - YHe))
        energy = 1.3806e-16 / 1.6726e-24 * Tini * npol / mu / unitv / unitv
        energies = np.full(original_header["ngas"], energy)
        f["PartType0/InternalEnergy"] = energies

        ic_params = f.create_group("ICs_parameters")
        lpt_params = ic_params.create_group("2LPT")
        lpt_params.attrs.update(read_2lpt())
        camb_params = ic_params.create_group("CAMB")
        camb_params.attrs.update(read_camb())

        # Make sure all particles are in the box.
        # Sometimes the conversion results in particles just barely outside the box.
        f["PartType0/Coordinates"][:] %= header["BoxSize"][0]
        f["PartType1/Coordinates"][:] %= header["BoxSize"][0]


def read_2lpt():
    result = {}
    with open("./2LPT.param") as f:
        while line := f.readline():
            try:
                name, value = line.split()
            except ValueError:
                continue
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass
            result[name] = value

    return result


def read_camb():
    result = {}
    group = None
    with open("./CAMB.params") as f:
        while line := f.readline():
            if ":" in line:
                # It's a group
                group = line.split(":")[0].strip()
            if line.startswith("   "):
                # It's part of a group
                try:
                    name, value = line.split(" = ")
                except ValueError:
                    continue
                assert type(group) is str
                name = group + ":" + name.strip()
                try:
                    value = ast.literal_eval(value)
                except ValueError:
                    pass
                if value is not None:
                    result[name] = value
            elif line.startswith(" "):
                try:
                    name, value = line.split(" = ")
                except ValueError:
                    continue
                name = name.strip()
                try:
                    value = ast.literal_eval(value)
                except ValueError:
                    pass
                if value is not None:
                    result[name] = value
    return result


if __name__ == "__main__":
    main()
