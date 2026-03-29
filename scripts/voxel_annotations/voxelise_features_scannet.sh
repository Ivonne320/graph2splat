#!/bin/bash
#SBATCH -n 1
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=4:00:00
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
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"


export VLSG_SPACE=$(pwd)
export DATA_ROOT_DIR=/cluster/project/cvg/Shared_datasets/3RScan/
export SCRATCH=/cluster/scratch/wangyih/overfitting_dataset/pretrained
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"

source .venv/bin/activate

/cluster/home/wangyih/miniconda3/envs/graph2splat/bin/python preprocessing/voxel_anno/voxelise_features_scannet_scene.py \
    --config "preprocessing/voxel_anno/voxel_anno_scannet.yaml" \
    --model_dir "/cluster/project/cvg/Shared_datasets/3RScan/" \
    ${args[@]}
