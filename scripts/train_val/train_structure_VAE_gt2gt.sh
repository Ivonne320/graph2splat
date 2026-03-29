#!/bin/bash
#SBATCH -n 1
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --tmp=15G
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu 10G
#SBATCH -J train_vae_gt2gt
#SBATCH -o logs/train_vae_gt2gt/train_vae_gt2gt%j.out
#SBATCH -e logs/train_vae_gt2gt/train_vae_gt2gt%j.err
#SBATCH --mail-type=ALL
#SBATCH --account ls_polle



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
# export VLSG_TRAINING_OUT_DIR="$SCRATCH/training_structure_model/$timestamp"
# export VLSG_TRAINING_OUT_DIR="/cluster/scratch/wangyih/overfitting_dataset/pretrained/training_structure_model/2025-10-24_10_scenes_after_warm"
export VLSG_TRAINING_OUT_DIR="/cluster/scratch/wangyih/overfitting_dataset/pretrained/training_structure_model_gt2gt/2025-11-18_100scenes_gt2gt_less_lr"
# Initialize conda and activate the environment
source .venv/bin/activate

# Navigate to VLSG space
cd "$VLSG_SPACE" || { echo "Failed to change directory to $VLSG_SPACE"; exit 1; }
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"

# Run training script
/cluster/home/wangyih/miniconda3/envs/graph2splat/bin/python src/trainval/train_structure_models_bbox_head_gt2gt.py   --config scripts/train_val/train_structure_vae_gt2gt.yaml --log_steps 1 output_dir=\"$VLSG_TRAINING_OUT_DIR\" ${args[@]}
