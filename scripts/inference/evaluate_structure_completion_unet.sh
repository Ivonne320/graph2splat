#!/bin/bash
#SBATCH -n 1
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=24:00:00
#SBATCH --tmp=15G
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu 10G
#SBATCH -J eval_unet
#SBATCH -o logs/eval_unet/eval_unet%j.out
#SBATCH -e logs/eval_unet/eval_unet%j.err
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
export VLSG_TRAINING_OUT_DIR="/cluster/scratch/wangyih/overfitting_dataset/pretrained/training_unet_completion/2025-11-5_700_scenes_unet"
# Initialize conda and activate the environment
source .venv/bin/activate

# Navigate to VLSG space
cd "$VLSG_SPACE" || { echo "Failed to change directory to $VLSG_SPACE"; exit 1; }
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"

# Run training script
/cluster/home/wangyih/miniconda3/envs/graph2splat/bin/python src/inference/unet_completion_evaluation.py --config scripts/train_val/train_structure.yaml --checkpoint /cluster/scratch/wangyih/overfitting_dataset/pretrained/training_unet_completion/2025-11-5_300_scenes_unet_fix_compressor_align_dino/snapshots/epoch-18.pth.tar --split=test --batch_size=16 
