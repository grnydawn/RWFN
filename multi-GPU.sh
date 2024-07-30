#!/bin/bash
#SBATCH -A nwp501
#SBATCH -J FCNet
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err

module load cray-python/3.9.13.1
module load rocm/6.0.0
#module load amd-mixed/6.0.0
module load craype-accel-amd-gfx90a

TOP_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
CONFIG_FILE=${TOP_DIR}/config/AFNO.yaml
CONFIG='afno_backbone'
RUN_NUM=0

ulimit -n 65536

export HDF5_USE_FILE_LOCKING=FALSE

export NCCL_NET_GDR_LEVEL=PHB
#export NCCL_DEBUG=info
#export NCCL_PROTO=Simple
export NCCL_SOCKET_IFNAME=hsn

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

export MASTER_ADDR=$(hostname)

export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

# Create a virtual env.
#python3 -m venv .venv

source .venv/bin/activate

# Install the required packages
#pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
#pip install h5py matplotlib ruamel.yaml timm einops scipy torch-tb-profiler

if [[ -n "${SLURM_NNODES}" ]]; then
  NNODES=${SLURM_NNODES}
  NGPUS=$(echo ${SLURM_NNODES}*${SLURM_GPUS_ON_NODE} | bc)
else
  NNODES=1
  NGPUS=8
fi

echo "#### Running on $NGPUS GPUs of $NNODES nodes. ####"

set -x
srun -n ${NGPUS} -u \
    bash -c "
    source ${TOP_DIR}/export_DDP_vars.sh
    python ${TOP_DIR}/train_ddp.py --enable_amp --yaml_config=$CONFIG_FILE --config=$CONFIG --run_num=$RUN_NUM
    "
