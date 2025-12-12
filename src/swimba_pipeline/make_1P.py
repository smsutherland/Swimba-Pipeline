from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table

from .make_run import Params, make_run

ONEP_SEED = 67
NUM_RUNS = 5


def main():
    parameters_fname = Path(__file__).parent / "./data/1P_params.txt"
    oneP_tab = Table.read(parameters_fname, format="ascii.basic", delimiter=",")

    default_param_file = Path(__file__).parent / "./data/default.yml"
    with open(default_param_file, "r") as handle:
        base_parameters: Params = yaml.load(handle, Loader=yaml.Loader)

    if Path("./CosmoAstroSeed_SWIMBA_L25n256_1P.txt").exists():
        cosmoastroseed = Table.read("./CosmoAstroSeed_SWIMBA_L25n256_1P.txt", format="ascii.basic")
    else:
        cosmoastroseed = Table(
            names=["sim_name", *oneP_tab["ParamName"], "seed"],
            dtype=[str] + [float] * len(oneP_tab) + [int],
        )
    cosmoastroseed.add_index("sim_name")

    runs = []

    for i, param in enumerate(oneP_tab):
        if param["LogFlag"]:
            min = param["Fiducial"] / param["AbsMaxDiff"]
            max = param["Fiducial"] * param["AbsMaxDiff"]
            values = np.geomspace(min, max, NUM_RUNS)
        else:
            min = param["Fiducial"] - param["AbsMaxDiff"]
            max = param["Fiducial"] + param["AbsMaxDiff"]
            values = np.linspace(min, max, NUM_RUNS)

        for j, val in enumerate(values):
            j -= 2
            path = Path.cwd() / f"1P_p{i+1}_{'n'*(j<0)}{abs(j)}"
            if path.exists():
                continue  # don't overwrite exiting runs
            if j == 0 and i != 0:
                path.symlink_to("./1P_p1_0")
                continue  # don't do multiple fiducials

            if param["ParamName"] == "Gravity:softening":
                changes = {
                    "Gravity": {
                        "comoving_baryon_softening": val,
                        "max_physical_baryon_softening": val,
                        "comoving_DM_softening": val,
                        "max_physical_DM_softening": val,
                    }
                }
            elif param["ParamName"] == "Cosmology:Omega_cdm":
                changes = {
                    "Cosmology": {
                        "Omega_cdm": val,
                        "Omega_lambda": 1
                        - base_parameters["Cosmology"]["Omega_b"]
                        - val,
                    }
                }
            elif param["ParamName"] == "Cosmology:Omega_b":
                changes = {
                    "Cosmology": {
                        "Omega_b": val,
                        "Omega_cdm": 1
                        - base_parameters["Cosmology"]["Omega_lambda"]
                        - val,
                    }
                }
            else:
                param_name = param["ParamName"]
                group, name = param_name.split(":")
                changes = {group: {name: val}}
            if "ICs" in changes:
                changes["ICs"]["seed"] = ONEP_SEED
            else:
                changes["ICs"] = {"seed": ONEP_SEED}

            make_run(changes, path)
            runs.append(path)
            these_params = list(oneP_tab["Fiducial"])
            these_params[i] = val
            try:
                cosmoastroseed.loc[str(path.name)] = [
                    str(path.name),
                    *these_params,
                    ONEP_SEED,
                ]
            except KeyError:
                cosmoastroseed.add_row([str(path.name), *these_params, ONEP_SEED])

    cosmoastroseed.sort("sim_name")
    cosmoastroseed.write(
        "./CosmoAstroSeed_SWIMBA_L25n256_1P.txt", format="ascii.basic", overwrite=True
    )

    with open("1P.sh", "w") as f:
        f.write(f"""#!/bin/bash
#########################################################
#SBATCH --job-name=SWIMBA_CAMELS_1P
#SBATCH --partition=cmbas
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


if __name__ == "__main__":
    main()
