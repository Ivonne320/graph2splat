#!/bin/bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 256G
#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --time 3-00:00:00

args=("$@")

export VLSG_SPACE=$(pwd)
export PYTHONPATH="$VLSG_SPACE:$PYTHONPATH:$VLSG_SPACE/dependencies/gaussian-splatting"
export CONDA_BIN=$CONDA_PREFIX/bin
export DATA_ROOT_DIR="/work/vita/datasets/3RScan"

source /work/vita/yiwang/miniconda3/etc/profile.d/conda.sh
conda activate graph2splat

# source .venv/bin/activate
# Print GPU info at job start
echo "=== NVIDIA SMI at job start ==="
nvidia-smi
echo "==============================="

python preprocessing/voxel_anno/voxelise_features.py \
    --config "preprocessing/voxel_anno/voxel_anno.yaml" \
    --model_dir "$DATA_ROOT_DIR" \
    ${args[@]}
