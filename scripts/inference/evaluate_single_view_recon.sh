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
/cluster/home/wangyih/miniconda3/envs/graph2splat/bin/python src/inference/slat_completion_inference.py --config scripts/train_val/train_structure.yaml \
    --unet_checkpoint /cluster/scratch/wangyih/overfitting_dataset/pretrained/training_unet_slat_completion/student/500scenes_use_obj_filter/snapshots/epoch-2.pth.tar \
    --teacher_checkpoint /cluster/scratch/wangyih/overfitting_dataset/pretrained/debug_gs/teacher/200scenes-128/snapshots/epoch-264.pth.tar \
    --eval_split val --max_eval 10000 --student_pack_root /cluster/scratch/wangyih/3RScan  --scene_id 02b33dfb-be2b-2d54-92d2-cd012b2b3c40  --use_obj_id_filter  --mc_passes 5
    # --eval_other_views --eval_other_views_num 3
# ['02b33dfb-be2b-2d54-92d2-cd012b2b3c40', 'fcf66d9e-622d-291c-84c2-bb23dfe31327', '02b33df9-be2b-2d54-9062-1253be3ce186', '02b33dfd-be2b-2d54-91d2-55454852009e', '02b33e01-be2b-2d54-93fb-4145a709cec5']
# /cluster/scratch/wangyih/overfitting_dataset/pretrained/debug_gs/batch/debug/latent_16_scene_level_clamp_gaussian_scale_10scenes_clamp_0_9_shuffle/snapshots/epoch-440.pth.tar
# /cluster/scratch/wangyih/overfitting_dataset/pretrained/debug_gs/batch/debug/latent_16_scene_level_clamp_gaussian_scale_100scenes/snapshots/epoch-162.pth.tar