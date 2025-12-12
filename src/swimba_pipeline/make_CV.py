from pathlib import Path

from astropy.table import Table

from .make_run import make_run

SEEDS = [
    1,
    2,
    3999666,
    4,
    5,
    6,
    7,
    8,
    9999666,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18999666,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
]
NUM_RUNS = len(SEEDS)


def main():
    if Path("./CosmoAstroSeed_SWIMBA_L25n256_1P.txt").exists():
        cosmoastroseed = Table.read(
            "./CosmoAstroSeed_SWIMBA_L25n256_1P.txt", format="ascii.basic"
        )
    else:
        cosmoastroseed = Table(
            names=[
                "Name",
                "Omega_m",
                "sigma_8",
                "A_SN1",
                "A_AGN1",
                "A_SN2",
                "A_AGN2",
                "seed",
            ],
            dtype=[str] + [float] * 6 + [int],
        )
    cosmoastroseed.add_index("Name")

    for i in range(NUM_RUNS):
        seed = SEEDS[i]
        row = ["CV_" + str(i), 0.3, 0.8] + [1.0] * 4 + [seed]
        path = Path("CV_" + str(i))

        try:
            cosmoastroseed.loc[str(path.name)] = row
        except KeyError:
            cosmoastroseed.add_row(row)
        changes = {"ICs": {"seed": seed}}
        make_run(changes, path)

    cosmoastroseed.sort("Name")
    cosmoastroseed.write(
        "./CosmoAstroSeed_SWIMBA_L25n256_CV.txt", format="ascii.basic", overwrite=True
    )

    with open("CV.sh", "w") as f:
        f.write(
            f"""#!/bin/bash
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
#SBATCH --array=0-{NUM_RUNS-1}
#########################################################

set -e"""
            """
cd ${SLURM_ARRAY_TASK_ID}
./job.sh
"""
        )


if __name__ == "__main__":
    main()
