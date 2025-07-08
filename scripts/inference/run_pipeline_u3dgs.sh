#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 256G
#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --time 3-00:00:00


export VLSG_SPACE=/home/yiwang/graph2splat/dependencies/VLSG
export SCRATCH=/home/yiwang/graph2splat/pretrained/
export DATA_ROOT_DIR=/work/vita/datasets/3RScan/
export CONDA_BIN=$CONDA_PREFIX/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

source /work/vita/yiwang/miniconda3/etc/profile.d/conda.sh
conda activate graph2splat

args=("$@")

# Set environment variables
export VLSG_SPACE=$(pwd)
export RESUME_DIR="$VLSG_TRAINING_OUT_DIR"

# get output directory argument if it exists
for i in "$@"
do
case $i in
    -o=*|--output_dir=*)
    export VLSG_TRAINING_OUT_DIR="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

# Set output directory
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
export VLSG_TRAINING_OUT_DIR="$SCRATCH/test_latent_autoencoder/$timestamp"

# Initialize conda and activate the environment
# source .venv/bin/activate

# Navigate to VLSG space
cd "$VLSG_SPACE" || { echo "Failed to change directory to $VLSG_SPACE"; exit 1; }
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"

python src/inference/unstructured_latent_inference_gnn_cluster.py --config configs/config.yaml ${args[@]}
