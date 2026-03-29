#!/bin/bash
#SBATCH -n 1
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=24:00:00
#SBATCH --tmp=15G
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu 10G
#SBATCH -J voxelize_scene_features
#SBATCH -o logs/voxelize_scene_final/voxelize_scene_%j.out
#SBATCH -e logs/voxelize_scene_final/voxelize_scene_%j.err
#SBATCH --mail-type=ALL
#SBATCH --account ls_polle



module load stack/2024-06
module load cuda/12.1.1
module load eth_proxy

args=("$@")

# Environment setup
export VLSG_SPACE=$(pwd)
export DATA_ROOT_DIR=/cluster/project/cvg/Shared_datasets/3RScan/
export SCRATCH=/cluster/scratch/wangyih/overfitting_dataset/pretrained
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"

# source .venv/bin/activate
# /cluster/home/wangyih/miniconda3/envs/graph2splat/bin/python preprocessing/voxel_anno/voxelise_features_scene.py \
cmd=(
    /cluster/home/wangyih/miniconda3/envs/graph2splat/bin/python
    preprocessing/voxel_anno/voxelise_features_scene_clean.py
    --config "preprocessing/voxel_anno/voxel_anno.yaml"
    --split train
    --model_dir /cluster/project/cvg/Shared_datasets/3RScan/
)

cmd+=("${args[@]}")

"${cmd[@]}"
