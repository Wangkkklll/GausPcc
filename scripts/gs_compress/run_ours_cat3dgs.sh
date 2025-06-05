#!/bin/bash

# CAT-3DGS Training Script for Multiple Datasets
# Usage: ./run_all.sh [scene_name]
# If scene_name is provided, only that scene will be processed
# Otherwise, all scenes in each dataset will be processed

# Parse command line arguments
cd /src/gs_compress/Cat-3DGS
SCENE_ARG=${1:-""}

# Function to get all scenes in a dataset directory
get_scenes() {
    local dataset_path=$1
    if [ -n "$SCENE_ARG" ]; then
        echo "$SCENE_ARG"
    else
        find "$dataset_path" -maxdepth 1 -type d -exec basename {} \; | grep -v "$(basename "$dataset_path")" | sort
    fi
}

echo "=== CAT-3DGS Training Script Started ==="
if [ -n "$SCENE_ARG" ]; then
    echo "Processing specific scene: $SCENE_ARG"
else
    echo "Processing all scenes in all datasets"
fi

# MipNeRF360 Dataset Configuration
echo -e "\n=== Processing MipNeRF360 Dataset ===\n"
S_PTH_MIPNERF="/data/wkl/gszip_data/mip_nerf360"
PROJECT_PREFIX_MIPNERF="CAT3DGS_mipsnerf360"
TXT_PREFIX_MIPNERF="output_result/mipsnerf360"
M_PTH_MIPNERF="output_log/mipsnerf360"
VOXEL_SIZE_MIPNERF=0.001
CMD_EXTRA_MIPNERF="--chcm_slices_list 5 10 15 20"

# Get scenes for MipNeRF360
SCENES_MIPNERF=$(get_scenes "$S_PTH_MIPNERF")
echo "Will process the following scenes in MipNeRF360: $SCENES_MIPNERF"

# Process MipNeRF360 scenes
for scene in $SCENES_MIPNERF; do
    echo -e "\n=== Processing MipNeRF360 scene: $scene ==="
    
    # Lambda values: you can modify this list as needed
    # Options: 0.002, 0.004, 0.01, 0.015, 0.03, 0.04
    for lmbda in 0.001; do
        PROJECT_NAME="${PROJECT_PREFIX_MIPNERF}_${scene}"
        TXT_NAME="${TXT_PREFIX_MIPNERF}/${scene}"
        
        echo "Executing training with lambda=$lmbda for scene $scene"
        CUDA_VISIBLE_DEVICES=1 python train.py \
            --eval \
            --lod 0 \
            --voxel_size $VOXEL_SIZE_MIPNERF \
            --update_init_factor 16 \
            --iterations 40_000 \
            --test_iterations 40000 \
            -s "${S_PTH_MIPNERF}/${scene}" \
            -m "${M_PTH_MIPNERF}/${scene}/${lmbda}" \
            --exp_name "${scene}_${lmbda}" \
            --lmbda $lmbda \
            --project_name "$PROJECT_NAME" \
            --cam_mask 1 \
            --txt_name "$TXT_NAME" \
            $CMD_EXTRA_MIPNERF
    done
done

# Deep Blending Dataset Configuration
echo -e "\n=== Processing Deep Blending Dataset ===\n"
S_PTH_DB="/data/wkl/gszip_data/db"
PROJECT_PREFIX_DB="CAT3DGS_db"
TXT_PREFIX_DB="output_result/db"
M_PTH_DB="output_log/db"
VOXEL_SIZE_DB=0.005
CMD_EXTRA_DB="--chcm_for_offsets --chcm_for_scaling"

# Get scenes for Deep Blending
SCENES_DB=$(get_scenes "$S_PTH_DB")
echo "Will process the following scenes in Deep Blending: $SCENES_DB"

# Process Deep Blending scenes
for scene in $SCENES_DB; do
    echo -e "\n=== Processing Deep Blending scene: $scene ==="
    
    # Lambda values: you can modify this list as needed
    for lmbda in 0.001; do
        PROJECT_NAME="${PROJECT_PREFIX_DB}_${scene}"
        TXT_NAME="${TXT_PREFIX_DB}/${scene}"
        
        echo "Executing training with lambda=$lmbda for scene $scene"
        CUDA_VISIBLE_DEVICES=1 python train.py \
            --eval \
            --lod 0 \
            --voxel_size $VOXEL_SIZE_DB \
            --update_init_factor 16 \
            --iterations 40_000 \
            --test_iterations 40000 \
            -s "${S_PTH_DB}/${scene}" \
            -m "${M_PTH_DB}/${scene}/${lmbda}" \
            --exp_name "${scene}_${lmbda}" \
            --lmbda $lmbda \
            --project_name "$PROJECT_NAME" \
            --cam_mask 1 \
            --txt_name "$TXT_NAME" \
            $CMD_EXTRA_DB
    done
done

# Tanks and Temples Dataset Configuration
echo -e "\n=== Processing Tanks and Temples Dataset ===\n"
S_PTH_TANDT="/data/wkl/gszip_data/tandt"
PROJECT_PREFIX_TANDT="CAT3DGS_tnt"
TXT_PREFIX_TANDT="output_result/tnt"
M_PTH_TANDT="output_log/tnt"
VOXEL_SIZE_TANDT=0.01
CMD_EXTRA_TANDT="--chcm_slices_list 5 10 15 20"

# Get scenes for Tanks and Temples
SCENES_TANDT=$(get_scenes "$S_PTH_TANDT")
echo "Will process the following scenes in Tanks and Temples: $SCENES_TANDT"

# Process Tanks and Temples scenes
for scene in $SCENES_TANDT; do
    echo -e "\n=== Processing Tanks and Temples scene: $scene ==="
    
    # Lambda values: you can modify this list as needed
    for lmbda in 0.001; do
        PROJECT_NAME="${PROJECT_PREFIX_TANDT}_${scene}"
        TXT_NAME="${TXT_PREFIX_TANDT}/${scene}"
        
        echo "Executing training with lambda=$lmbda for scene $scene"
        CUDA_VISIBLE_DEVICES=1 python train.py \
            --eval \
            --lod 0 \
            --voxel_size $VOXEL_SIZE_TANDT \
            --update_init_factor 16 \
            --iterations 40_000 \
            --test_iterations 40000 \
            -s "${S_PTH_TANDT}/${scene}" \
            -m "${M_PTH_TANDT}/${scene}/${lmbda}" \
            --exp_name "${scene}_${lmbda}" \
            --lmbda $lmbda \
            --project_name "$PROJECT_NAME" \
            --cam_mask 1 \
            --txt_name "$TXT_NAME" \
            $CMD_EXTRA_TANDT
    done
done

echo -e "\n=== All CAT-3DGS training tasks completed ==="