#!/bin/bash
#SBATCH -n 1
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=24:00:00
#SBATCH --tmp=15G
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu 10G
#SBATCH -J warm_up
#SBATCH -o logs/train_structure_model_warmup/train_structure_model%j.out
#SBATCH -e logs/train_structure_model_warmup/train_structure_model%j.err
#SBATCH --mail-type=ALL




module load stack/2024-06
module load cuda/12.1.1
module load eth_proxy


args=("$@")

# Set environment variables
export VLSG_SPACE=$(pwd)
export RESUME_DIR="$VLSG_TRAINING_OUT_DIR"
export DATA_ROOT_DIR=/cluster/project/cvg/Shared_datasets/3RScan/
export SCRATCH=/cluster/scratch/wangyih/overfitting_dataset/pretrained
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"

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
# export VLSG_TRAINING_OUT_DIR="$SCRATCH/training_structure_model_warm_up/$timestamp"
export VLSG_TRAINING_OUT_DIR="/cluster/scratch/wangyih/overfitting_dataset/pretrained/training_structure_model_warm_up/20251028_700scenes_fix_batch"
# Initialize conda and activate the environment
source .venv/bin/activate

# Navigate to VLSG space
cd "$VLSG_SPACE" || { echo "Failed to change directory to $VLSG_SPACE"; exit 1; }
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"

# Run training script
/cluster/home/wangyih/miniconda3/envs/graph2splat/bin/python src/trainval/train_warm_up_structure_models_bbox_head.py  --config scripts/train_val/train_structure_warmup.yaml  --log_steps 1 output_dir=\"$VLSG_TRAINING_OUT_DIR\" ${args[@]}
