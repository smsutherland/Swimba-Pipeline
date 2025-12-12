from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table
from swiftemulator import ModelSpecification
from swiftemulator.design import latin

from .make_run import Params, make_run

STARTING_SEED = 50000
np.random.seed(9)


def loop(row):
    target = Path(row["sim_name"])
    if target.exists():
        return  # don't overwrite exiting runs
    parameters = {}
    for param in row.colnames:
        if param == "sim_name":
            continue
        val = row[param]
        if param == "Gravity:softening":
            if "Gravity" not in parameters:
                parameters["Gravity"] = {}
            parameters["Gravity"].update(
                {
                    "comoving_baryon_softening": val,
                    "max_physical_baryon_softening": val,
                    "comoving_DM_softening": val,
                    "max_physical_DM_softening": val,
                }
            )
        elif param == "seed":
            if "ICs" not in parameters:
                parameters["ICs"] = {}
            parameters["ICs"]["seed"] = val
        else:
            group, name = param.split(":")
            if group not in parameters:
                parameters[group] = {}
            parameters[group][name] = val

    default_param_file = Path(__file__).parent / "./data/default.yml"
    with open(default_param_file, "r") as handle:
        base_parameters: Params = yaml.load(handle, Loader=yaml.Loader)

    # Fix cosmology
    if "Cosmology" in parameters:
        cosmo = parameters["Cosmology"]
        o_cdm = cosmo.get("Omega_cdm") or base_parameters["Cosmology"]["Omega_cdm"]
        o_b = cosmo.get("Omega_b") or base_parameters["Cosmology"]["Omega_b"]
        cosmo["Omega_lambda"] = 1 - o_cdm - o_b

    make_run(parameters, target=target)
    return target


def main():
    parameters_fname = Path(__file__).parent / "./data/1P_params.txt"
    parameters = Table.read(parameters_fname, format="ascii.basic", delimiter=",")[:6]

    if not Path("./CosmoAstroSeed_SWIMBA_L25n256_LH.txt").exists():
        cosmoastroseed = make_parameters(parameters)
    else:
        cosmoastroseed = Table.read(
            "./CosmoAstroSeed_SWIMBA_L25n256_LH.txt", format="ascii.basic"
        )
    cosmoastroseed.add_index("sim_name")

    with Pool() as p:
        runs = [x for x in p.map(loop, cosmoastroseed) if x is not None]

    with open("LH.sh", "w") as f:
        f.write(f"""#!/bin/bash
#########################################################
#SBATCH --job-name=SWIMBA_CAMELS_LH
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=cpu
#SBATCH --mail-user=sutherland.sagan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="logs/slurm-%A_%a.out"
#########################################################
#SBATCH --time=7-0
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=200G
#########################################################
#SBATCH --array=0-{len(runs) - 1}
#########################################################

set -e

sim_dirs=(
""")
        for run in runs:
            f.write(f'"{run.name}"\n')
        f.write(""")

cd ${sim_dirs[$SLURM_ARRAY_TASK_ID]}
./job.sh
""")


def make_parameters(parameters):
    parameter_names = parameters["ParamName"].tolist()
    parameter_limits = []
    parameter_transforms = {}
    for param in parameters:
        fid = param["Fiducial"]
        diff = param["AbsMaxDiff"]
        if param["LogFlag"]:
            lower = np.log10(fid / diff)
            upper = np.log10(fid * diff)
            parameter_transforms[param["ParamName"]] = lambda x: 10.0**x
        else:
            lower = fid - diff
            upper = fid + diff
        parameter_limits.append([lower, upper])

    spec = ModelSpecification(
        number_of_parameters=6,
        parameter_names=parameter_names,
        parameter_limits=parameter_limits,
    )

    number_of_simulations = 1000

    model_parameters = latin.create_hypercube(
        model_specification=spec,
        number_of_samples=number_of_simulations,
        prefix_unique_id="LH_",
    )

    cosmoastroseed = Table(
        names=["sim_name", *parameters["ParamName"], "seed"],
        dtype=[str] + [float] * len(parameters) + [int],
    )
    seed = STARTING_SEED
    for name, params in model_parameters.items():
        row = {"sim_name": str(name), **params, "seed": seed}
        for k, f in parameter_transforms.items():
            row[k] = f(row[k])  # Un-log relevant parameters
        cosmoastroseed.add_row(row)
        seed += 1

    cosmoastroseed.write("./CosmoAstroSeed_SWIMBA_L25n256_LH.txt", format="ascii.basic")
    return cosmoastroseed


if __name__ == "__main__":
    main()
