import argparse
import shutil
from glob import glob
from pathlib import Path
import concurrent.futures
from astropy.table import Table

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    force = args.force
    target = source.parent.parent.parent / (source.relative_to(source.parent.parent))

    (cosmoastroseed,) = glob(str(source / "CosmoAstroSeed*"))
    cosmoastroseed = Path(cosmoastroseed).resolve()
    shutil.copy(cosmoastroseed, target / cosmoastroseed.name)
    cosmoastroseed_table: Table = Table.read(cosmoastroseed, format="ascii.basic")
    cosmoastroseed_table.add_index(cosmoastroseed_table.colnames[0])

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as p:
        children = list(source.iterdir())
        futures = {
            p.submit(iter, child, target, force, cosmoastroseed_table): child
            for child in children
        }
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(children)):
            pass


def iter(child: Path, target: Path, force: bool, cosmoastroseed_table: Table):
    if child.is_dir() and child.name != "logs" and not child.is_symlink():
        tqdm.write(f"{child.name}")
        if (target / child.name).exists():
            if force:
                shutil.rmtree(target / child.name)
            else:
                tqdm.write(f"WARNING: skipping {child.name}! Use --force to overwrite.")
                return
        data_files(child, target / child.name)
        ICs(child, target / child.name)
        extra_files(child, target / child.name)
        cosmoastroseed_file(target / child.name, cosmoastroseed_table)


def data_files(source_dir: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    snaps = source_dir / "snaps"
    for i in range(0, 91):
        snap = snaps / f"snapshot_{i:04}.hdf5"
        target_snap = target_dir / f"snapshot_{i:03}.hdf5"
        target_snap.hardlink_to(snap)
    subs = snaps / "subs"
    for i in range(0, 91):
        fof = subs / f"fof_subhalo_tab_{i:03}.hdf5"
        target_fof = target_dir / f"groups_{i:03}.hdf5"
        target_fof.hardlink_to(fof)


def ICs(source_dir: Path, target_dir: Path):
    target_dir /= "ICs"
    target_dir.mkdir()
    (target_dir / "ic.hdf5").hardlink_to(source_dir / "ic.hdf5")
    source_dir /= "ICs"
    copy_files = ["2LPT.param", "CAMB.params", "inputspec_ics.txt", "Pk_m_z=0.000.txt"]
    for f in copy_files:
        shutil.copy(source_dir / f, target_dir / f)


def extra_files(source_dir: Path, target_dir: Path):
    target_dir /= "extra_files"
    target_dir.mkdir()
    files = [
        "dependency_graph_0.csv",
        "output.yml",
        "output_list.txt",
        "params.yml",
        "SFR.txt",
        "snapshot.xmf",
        "statistics.txt",
        "swift.err",
        "swift.log",
        "task_level_0000_0.txt",
        "timesteps.txt",
        "unused_parameters.yml",
        "used_parameters.yml",
    ]
    for f in files:
        shutil.copy(source_dir / f, target_dir / f)


def cosmoastroseed_file(target_dir: Path, cosmoastroseed_table: Table):
    sim_name = target_dir.name
    row = cosmoastroseed_table.loc[sim_name]
    with open(target_dir / "CosmoAstro_params.txt", "w") as f:
        for item in row[1:]: # skip run name
            f.write(str(item))


if __name__ == "__main__":
    main()
