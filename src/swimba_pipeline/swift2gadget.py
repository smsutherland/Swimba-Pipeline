import argparse
import typing as T
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input snapshot filename.",
    )

    parser.add_argument(
        "output_file",
        type=Path,
        help="Output snapshot filename.",
    )

    parser.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args()

    verbose: bool = args.verbose > 0

    out_snap: Path = args.output_file
    in_snap: Path = args.input_file

    if verbose:
        print(f"Input file: {in_snap}\n" f"Output file: {out_snap}")

    output_units = {
        "UnitLength_in_cm": 3.085678e21,
        "UnitMass_in_g": 1.989e43,
        "UnitVelocity_in_cm_per_s": 100000.0,
        "Msun_in_g": 1.989e33,
        "Year_in_s": 31557600,
    }

    output_units["UnitTime_in_s"] = (
        output_units["UnitLength_in_cm"] / output_units["UnitVelocity_in_cm_per_s"]
    )

    attrs_update = [
        "CanHaveTypes",
        "Flag_Entropy_ICs",
        "InitialMassTable",
        "MassTable",
        "NumPart_ThisFile",
        "NumPart_Total",
        "NumPart_Total_HighWord",
        "TotalNumberOfParticles",
    ]

    if out_snap.exists():
        if verbose:
            print("Deleting previous output file...")

        out_snap.unlink()

    with h5py.File(out_snap, "a") as hf_out:
        hf_out.create_group("Header")

        with h5py.File(in_snap, "r") as hf_in:
            scale_factor: float = hf_in["Cosmology"].attrs["Scale-factor"][0]
            hubble_param: float = hf_in["Cosmology"].attrs["H0 [internal units]"] / 100

            fields: list[tuple[str, str, T.Optional[float], bool]] = [
                ("PartType0/ParticleIDs", "PartType0/ParticleIDs", None, False),
                (
                    "PartType0/Coordinates",
                    "PartType0/Coordinates",
                    output_units["UnitLength_in_cm"] * scale_factor / hubble_param,
                    False,
                ),
                (
                    "PartType0/StarFormationRates",
                    "PartType0/StarFormationRate",
                    output_units["Msun_in_g"] / output_units["Year_in_s"],
                    True,
                ),
                (
                    "PartType0/Masses",
                    "PartType0/Masses",
                    output_units["UnitMass_in_g"] / hubble_param,
                    False,
                ),
                (
                    "PartType0/InternalEnergies",
                    "PartType0/InternalEnergy",
                    output_units["UnitVelocity_in_cm_per_s"] ** 2,
                    False,
                ),
                (
                    "PartType0/Densities",
                    "PartType0/Density",
                    output_units["UnitMass_in_g"]
                    * output_units["UnitLength_in_cm"] ** -3
                    * scale_factor**-3
                    * hubble_param**2,
                    False,
                ),
                (
                    "PartType0/Velocities",
                    "PartType0/Velocities",
                    output_units["UnitVelocity_in_cm_per_s"] * scale_factor**0.5,
                    False,
                ),
                (
                    "PartType0/SmoothingLengths",
                    "PartType0/SmoothingLength",
                    output_units["UnitLength_in_cm"] * scale_factor / hubble_param,
                    False,
                ),
                ("PartType0/Temperatures", "PartType0/Temperatures", 1, False),
                ("PartType0/ElectronNumberDensities", "PartType0/ElectronAbundance", 1, False),
                ("PartType1/ParticleIDs", "PartType1/ParticleIDs", None, False),
                (
                    "PartType1/Coordinates",
                    "PartType1/Coordinates",
                    output_units["UnitLength_in_cm"] * scale_factor / hubble_param,
                    False,
                ),
                (
                    "PartType1/Masses",
                    "PartType1/Masses",
                    output_units["UnitMass_in_g"] / hubble_param,
                    False,
                ),
                (
                    "PartType1/Velocities",
                    "PartType1/Velocities",
                    output_units["UnitVelocity_in_cm_per_s"] * scale_factor**0.5,
                    False,
                ),
                ("PartType4/ParticleIDs", "PartType4/ParticleIDs", None, False),
                (
                    "PartType4/Coordinates",
                    "PartType4/Coordinates",
                    output_units["UnitLength_in_cm"] * scale_factor / hubble_param,
                    False,
                ),
                (
                    "PartType4/Masses",
                    "PartType4/Masses",
                    output_units["UnitMass_in_g"] / hubble_param,
                    False,
                ),
                (
                    "PartType4/Velocities",
                    "PartType4/Velocities",
                    output_units["UnitVelocity_in_cm_per_s"] * scale_factor**0.5,
                    False,
                ),
                (
                    "PartType4/SmoothingLengths",
                    "PartType4/SmoothingLength",
                    output_units["UnitLength_in_cm"] * scale_factor / hubble_param,
                    False,
                ),
                (
                    "PartType4/InitialMasses",
                    "PartType4/InitialMass",
                    output_units["UnitMass_in_g"] / hubble_param,
                    False,
                ),
                (
                    "PartType4/BirthScaleFactors",
                    "PartType4/StellarFormationTime",
                    None,
                    False,
                ),
                ("PartType5/ParticleIDs", "PartType5/ParticleIDs", None, False),
                (
                    "PartType5/Coordinates",
                    "PartType5/Coordinates",
                    output_units["UnitLength_in_cm"] * scale_factor / hubble_param,
                    False,
                ),
                (
                    "PartType5/DynamicalMasses",
                    "PartType5/Masses",
                    output_units["UnitMass_in_g"] / hubble_param,
                    False,
                ),
                (
                    "PartType5/SubgridMasses",
                    "PartType5/BH_Mass",
                    output_units["UnitMass_in_g"] / hubble_param,
                    False,
                ),
                (
                    "PartType5/Velocities",
                    "PartType5/Velocities",
                    output_units["UnitVelocity_in_cm_per_s"] * scale_factor**0.5,
                    False,
                ),
                (
                    "PartType5/SmoothingLengths",
                    "PartType5/SmoothingLength",
                    output_units["UnitLength_in_cm"] * scale_factor / hubble_param,
                    False,
                ),
                (
                    "PartType5/AccretionRates",
                    "PartType5/Mdot",
                    output_units["UnitMass_in_g"] / output_units["UnitTime_in_s"],
                    False,
                ),
            ]

            ## set up input units
            # input_units = {
            #     'UnitLength_in_cm': 3.08568e+24,
            #     'UnitMass_in_g': 1.98841e+43,
            #     'UnitTime_in_s': 3.08568e+19,

            # }
            # input_units['UnitVelocity_in_cm_per_s'] = input_units['UnitLength_in_cm'] /\
            #         input_units['UnitTime_in_s']

            hf_out["Header"].attrs.update(hf_in["Header"].attrs)

            # ## add cosmo information
            hf_out["Header"].attrs["Omega0"] = (
                hf_in["Cosmology"].attrs["Omega_b"][0]
                + hf_in["Cosmology"].attrs["Omega_cdm"][0]
            )
            hf_out["Header"].attrs["OmegaLambda"] = hf_in["Cosmology"].attrs[
                "Omega_lambda"
            ][0]
            hf_out["Header"].attrs["HubbleParam"] = hf_in["Cosmology"].attrs["h"][0]

            ## add extra flags
            hf_out["Header"].attrs["Flag_Cooling"] = 0
            hf_out["Header"].attrs["Flag_DoublePrecision"] = 0
            hf_out["Header"].attrs["Flag_Feedback"] = 0
            hf_out["Header"].attrs["Flag_IC_Info"] = 0
            hf_out["Header"].attrs["Flag_Metals"] = 11
            hf_out["Header"].attrs["Flag_Sfr"] = 0
            hf_out["Header"].attrs["Flag_StellarAge"] = 0

            ## fix boxsize dimensions (3 -> 1)
            hf_out["Header"].attrs["BoxSize"] = (
                hf_out["Header"].attrs["BoxSize"][0] * hubble_param[0] * 1000
            )

            hf_out["Header"].attrs["Redshift"] = hf_out["Header"].attrs["Redshift"][0]
            hf_out["Header"].attrs["Time"] = hf_out["Header"].attrs["Scale-factor"][0]

            hf_out["Header"].attrs["ExpansionFactor"] = 1.0 / (
                1.0 + hf_out["Header"].attrs["Redshift"]
            )

            # Omega_0 = hf_out['Header'].attrs['Omega0']
            # redshift = hf_out['Header'].attrs['Redshift']
            # Omega_l = hf_out['Header'].attrs['OmegaLambda']
            # hf_out['Header'].attrs['H(z)'] = 100.0*np.sqrt(Omega_0*(1.0+redshift)**3+Omega_l)

            ## Remove extra PartType (7 -> 6) from header info
            hf_out["Header"].attrs["NumPartTypes"] = 6

            for attr in attrs_update:
                hf_out["Header"].attrs[attr] = hf_out["Header"].attrs[attr][:-1]

            ## Time means time since big bang in SWIFT, but scale_factor in GADGET
            hf_out["Header"].attrs["Time"] = scale_factor

            for field in fields:
                in_name, out_name, out_unit, remove_negative = field
                if verbose:
                    print(f"Writing {field}")

                temp = hf_in[in_name][:]

                ## get SWIFT conversions to CGS
                conv_factor = hf_in[in_name].attrs[
                    (
                        "Conversion factor to physical CGS "
                        "(including cosmological corrections)"
                    )
                ]

                # remove negative values (applicable for SFRs)
                if remove_negative:
                    temp[temp < 0] = 0.0

                if (conv_factor != 1.0) and (out_unit is not None):
                    conv_factor /= out_unit
                    temp *= conv_factor

                    ## Convert CGS to GADGET units
                    hf_out[out_name] = temp  # / field[2]
                else:
                    hf_out[out_name] = temp

            # Do metallicities separately
            in_fields = [
                "PartType0/ElementMassFractions",
                "PartType4/ElementMassFractions",
            ]
            out_fields = [
                "PartType0/Metallicity",
                "PartType4/Metallicity",
            ]

            for in_field, out_field in zip(in_fields, out_fields):
                temp = hf_in[in_field][:]

                out = np.zeros((len(temp), 9))

                out[:, 0] = np.sum(temp[:, 2:], axis=1)

                out[:, 1:] = temp[:, 1:]

                hf_out[out_field] = out

            hf_out["PartType0/NeutralHydrogenAbundance"] = hf_in["PartType0/SpeciesFractions"][:, 0]


if __name__ == "__main__":
    main()
