from scipy.stats.qmc import Sobol
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table
from swiftemulator import ModelSpecification
from swiftemulator.design import latin

from .make_run import Params, make_run

STARTING_SEED = 55000
SOBOL_SEED = 12
LOG_NUM_RUNS = 10  # 2**10 = 1024


def main():
    parameters_fname = Path(__file__).parent / "./data/1P_params.txt"
    parameters = Table.read(parameters_fname, format="ascii.basic", delimiter=",")[:6]
    if not Path("./CosmoAstroSeed_SWIMBA_L25n256_SB28.txt").exists():
        cosmoastroseed = make_parameters(parameters)
    else:
        cosmoastroseed = Table.read(
            "./CosmoAstroSeed_SWIMBA_L25n256_SB28.txt", format="ascii.basic"
        )
    cosmoastroseed.add_index("sim_name")

    with Pool() as p:
        runs = [x for x in p.map(loop, cosmoastroseed) if x is not None]

    with open("SB28.sh", "w") as f:
        f.write(
            f"""#!/bin/bash
#########################################################
#SBATCH --job-name=SWIMBA_CAMELS_SB28
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

set -e"""
            """
cd SB28_${SLURM_ARRAY_TASK_ID}
./job.sh
"""
        )


def make_parameters(parameters):
    parameter_limits = []
    parameter_transforms = {}
    for param in parameters:
        fid = param["Fiducial"]
        diff = param["AbsMaxDiff"]
        if param["LogFlag"]:
            lower = fid / diff
            upper = fid * diff
            parameter_transforms[param["ParamName"]] = lambda x: upper**x * lower ** (
                1 - x
            )
        else:
            lower = fid - diff
            upper = fid + diff
            parameter_transforms[param["ParamName"]] = (
                lambda x: x * upper + (1 - x) * lower
            )
        parameter_limits.append([lower, upper])

    sobol = Sobol(len(parameters), scramble=True, seed=SOBOL_SEED)
    params = sobol.random_base2(LOG_NUM_RUNS)

    cosmoastroseed = Table(
        names=["sim_name", *parameters["ParamName"], "seed"],
        dtype=[str] + [float] * len(parameters) + [int],
    )
    seed = STARTING_SEED
    for i, values in enumerate(params):
        row = {name: value for name, value in zip(parameters["ParamName"], values)}
        row["sim_name"] = "SB28_" + str(i)
        row["seed"] = seed
        for k, f in parameter_transforms.items():
            row[k] = f(row[k])
        cosmoastroseed.add_row(row)
        seed += 1

    cosmoastroseed.write("./CosmoAstroSeed_SWIMBA_L25n256_LH.txt", format="ascii.basic")
    return cosmoastroseed


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


if __name__ == "__main__":
    main()
