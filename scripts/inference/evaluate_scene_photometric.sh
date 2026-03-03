#!/bin/bash
#SBATCH -n 1
#SBATCH --gpus=a100-pcie-40gb:1
#SBATCH --time=24:00:00
#SBATCH --tmp=15G
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu 10G
#SBATCH -J train_gs_inpaint
#SBATCH -o logs/train_gs_inpaint/train_gs_inpaint%j.out
#SBATCH -e logs/train_gs_inpaint/train_gs_inpaint%j.err
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
# export VLSG_TRAINING_OUT_DIR="$SCRATCH/training_scene_decoder_scene_level_whole_set/$timestamp"
export VLSG_TRAINING_OUT_DIR="$SCRATCH/debug_gs/gs_inpaint/10_scene_20251207"
# export VLSG_TRAINING_OUT_DIR="/cluster/scratch/wangyih/overfitting_dataset/pretrained/training_scene_decoder_scene_level_whole_set/2025-10-12_00-21-50"
# Initialize conda and activate the environment
source .venv/bin/activate

# Navigate to VLSG space
cd "$VLSG_SPACE" || { echo "Failed to change directory to $VLSG_SPACE"; exit 1; }
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"
# Run training script
/cluster/home/wangyih/miniconda3/envs/graph2splat/bin/python src/inference/evaluate_scene_photometric.py --config scripts/train_val/train_batch_scene_gs.yaml   --checkpoint /cluster/scratch/wangyih/overfitting_dataset/pretrained/debug_gs/10-clean/test-256-decoder-only/snapshots/epoch-1080.pth.tar --split train --max_frames 60 --batch_size 16 ${args[@]}
