#!/bin/bash -l
#########################################################
#SBATCH --job-name=SWIMBA_CAMELS
#SBATCH --partition=cmbas
#SBATCH --constraint=cpu
#SBATCH --mail-user=sutherland.sagan@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="slurm-%A.out"
#########################################################
#SBATCH --time=7-0
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#########################################################

function swift_mods() {
    module --force purge
    module load modules/2.3-20240529
    module load openblas/single-0.3.26
    module load gcc/11.4.0
    module load openmpi/4.0.7
    module load hdf5/1.12.3
    module load gsl/2.7.1
    module load fftw/mpi-3.3.10
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/mnt/home/ssutherland/codes/libs/parmetis/lib/"
}

function arepo_mods() {
    module --force purge
    module load modules/2.0-20220630
    module load slurm
    module load disBatch/2.0
    module load gcc/7.5.0
    module load openmpi/4.0.7
    module load hdf5/mpi-1.10.8
    module load gmp/6.2.1
    module load fftw/3.3.10
    module load openblas/threaded-0.3.20
    module load gsl/2.7
    module load hwloc/2.7.1
}

function 2lpt_mods() {
    module --force purge
    module load modules/2.0-20220630
    module load gcc/11.2.0
    module load openmpi/1.10.7
    module load fftw/mpi-2.1.5
    module load gsl/2.7
}

function get_host() {
    case $(hostname) in
        worker5*)
            echo rome
            ;;
        worker6*)
            echo icelake
            ;;
        worker7*)
            echo genoa
            ;;
    esac
}

set -e

# IC generation
if [ ! -e ic.hdf5 ]; then
    pushd ICs
    2lpt_mods
    pushd 2lpt
    make clean
    make -j
    popd

    srun --ntasks=${SLURM_NTASKS} --cpus-per-task=1 ./2lpt/2LPTic 2LPT.param > 2lpt.log 2> 2lpt.err
    ./convert_ic.py ./ics ../ic.hdf5
    popd
fi

# SWIFT
swift_mods

# Build swift if it doesn't exist, or if it was built for a different kind of node.
if [ ! -e swiftsim/swift_mpi ] || [ "x$(get_host)" != "x$(cat host 2>/dev/null)" ]; then
    pushd swiftsim
    GRACKLE=/mnt/home/ssutherland/codes/libs/grackle
    PARMETIS=/mnt/home/ssutherland/codes/libs/parmetis/
    ONETBB=/mnt/home/ssutherland/codes/libs/oneTBB/
    ./autogen.sh
    CC=mpicc ./configure --with-subgrid=SAGAN --with-hydro=sphenix --with-hdf5=`which h5cc` --with-grackle=${GRACKLE} --with-parmetis=${PARMETIS} --with-tbbmalloc=${ONETBB} --with-gcc-arch=native
    make -j

    get_host > host

    popd
fi


# Has the simulation already run?
if [ ! -e snaps/snapshot_0090.hdf5 ]; then
    # It hasn't! Let's run it!

    if [ -e restart/swift_000000.rst ]; then
        # If a restart file exists, we probably want to be using it
        RESTART="--restart"
    fi

    echo job id: $SLURM_JOBID | tee -a swift.log
    if [[ $SLURM_ARRAY_TASK_ID ]]; then 
        echo job array id: $SLURM_ARRAY_TASK_ID | tee -a swift.log
    fi

    srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_ON_NODE} --cpu_bind=cores --kill-on-bad-exit=1 ./swift_mpi --pin --cosmology --simba --threads=${SLURM_CPUS_ON_NODE} ${RESTART} params.yml >> swift.log 2>> swift.err

    # Did we finish?
    if [ ! -e snapshot_0090.hdf5 ]; then
        echo SWIFT did not finish. Skipping subfind
        exit 12
    fi

    mkdir -p snaps/
    mv snapshot_*.hdf5 snaps/
fi

# SUBFIND
arepo_mods
mkdir -p ./snaps/subs/
source /mnt/home/ssutherland/.virtualenvs/sci/bin/activate
/mnt/home/ssutherland/codes/CAMELS-SWIFT/arepo_subfind/generate_params.py ./params.yml /mnt/home/ssutherland/codes/CAMELS-SWIFT/arepo_subfind/arepo_subfind_param_rusty.txt ./snaps/subs/arepo_subfind_param.txt

# Iterate backward through the snapshots
# This is done so that, during testing, snapshot 90 gets done first and we can look at it without having to wait
for i in $(seq 90 -1 0); do
    SOURCE_SNAP=$(printf "snapshot_%04d.hdf5" $i)
    DEST_SNAP=$(printf "subsnap_%03d.hdf5" $i)
    DEST_FOF=$(printf "fof_subhalo_tab_%03d.hdf5" $i)

    # Does the subfind catalog already exist?
    if [ -e snaps/subs/$DEST_FOF ]; then
        # Skip it!
        continue
    fi

    /mnt/home/ssutherland/codes/CAMELS-SWIFT/arepo_subfind/convert_snap_for_subfind.py "snaps/$SOURCE_SNAP" ./snaps/subs/ -v --output-filename

    pushd ./snaps/subs
    srun --ntasks=${SLURM_NTASKS} --cpus-per-task=1 --cpu_bind=cores --hint=compute_bound --kill-on-bad-exit=1 /mnt/home/ssutherland/codes/Arepo_subfind_v2/Arepo ./arepo_subfind_param.txt 3 $i 2>&1 | tee "subfind_${i}.log"
    popd
done

# clean up everything we don't need anymore
rm -rf restart/
rm -f snaps/subs/subsnap*.hdf5
