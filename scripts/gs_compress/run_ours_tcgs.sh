#!/bin/bash

# TC-GS Training Script for Multiple Datasets
# Usage: ./run_train_all.sh

# Basic parameters
cd /src/gs_compress/TC-GS
GPU_ID=6
ITERATIONS=30000

echo "=== TC-GS Training Script Started ==="
echo "Using GPU: $GPU_ID"
echo "Iterations: $ITERATIONS"

# MipNeRF360 Dataset Configuration
echo -e "\n=== Processing MipNeRF360 Dataset ===\n"

# MipNeRF360 parameters
MIPNERF360_VOXEL_SIZE=0.001
MIPNERF360_UPDATE_INIT_FACTOR=16
MIPNERF360_APPEARANCE_DIM=0
MIPNERF360_RATIO=1
MIPNERF360_TRI_RESOLUTION=32
MIPNERF360_FEAT_DIM=50

# MipNeRF360 scenes
MIPNERF360_SCENES="bicycle garden stump room counter kitchen bonsai flowers treehill"

for scene in $MIPNERF360_SCENES; do
    echo "Processing MipNeRF360 scene: $scene"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        --eval \
        -s /public/DATA/wkl/3d_edit_dataset/mip_nerf360/$scene \
        --lod 0 \
        --gpu -1 \
        --voxel_size $MIPNERF360_VOXEL_SIZE \
        --update_init_factor $MIPNERF360_UPDATE_INIT_FACTOR \
        --appearance_dim $MIPNERF360_APPEARANCE_DIM \
        --ratio $MIPNERF360_RATIO \
        --iterations $ITERATIONS \
        --tri_resolution $MIPNERF360_TRI_RESOLUTION \
        --feat_dim $MIPNERF360_FEAT_DIM \
        -m outputs/mipnerf360/$scene
    
    echo "Completed MipNeRF360 scene: $scene"
done

# Deep Blending Dataset Configuration
echo -e "\n=== Processing Deep Blending Dataset ===\n"

# DB parameters
DB_VOXEL_SIZE=0.005
DB_UPDATE_INIT_FACTOR=16
DB_APPEARANCE_DIM=0
DB_RATIO=1
DB_TRI_RESOLUTION=16
DB_FEAT_DIM=50

# DB scenes
DB_SCENES="playroom drjohnson"

for scene in $DB_SCENES; do
    echo "Processing Deep Blending scene: $scene"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        --eval \
        -s /public/DATA/wkl/3d_edit_dataset/db/$scene \
        --lod 0 \
        --gpu -1 \
        --voxel_size $DB_VOXEL_SIZE \
        --update_init_factor $DB_UPDATE_INIT_FACTOR \
        --appearance_dim $DB_APPEARANCE_DIM \
        --ratio $DB_RATIO \
        --iterations $ITERATIONS \
        --tri_resolution $DB_TRI_RESOLUTION \
        --feat_dim $DB_FEAT_DIM \
        -m outputs/blending/$scene
    
    echo "Completed Deep Blending scene: $scene"
done

# Tanks and Temples Dataset Configuration
echo -e "\n=== Processing Tanks and Temples Dataset ===\n"

# TandT parameters
TANDT_VOXEL_SIZE=0.01
TANDT_UPDATE_INIT_FACTOR=16
TANDT_APPEARANCE_DIM=0
TANDT_RATIO=1
TANDT_TRI_RESOLUTION=16
TANDT_FEAT_DIM=50

# TandT scenes
TANDT_SCENES="truck train"

for scene in $TANDT_SCENES; do
    echo "Processing Tanks and Temples scene: $scene"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        --eval \
        -s /public/DATA/wkl/3d_edit_dataset/tandt/$scene \
        --lod 0 \
        --gpu -1 \
        --voxel_size $TANDT_VOXEL_SIZE \
        --update_init_factor $TANDT_UPDATE_INIT_FACTOR \
        --appearance_dim $TANDT_APPEARANCE_DIM \
        --ratio $TANDT_RATIO \
        --iterations $ITERATIONS \
        --tri_resolution $TANDT_TRI_RESOLUTION \
        --feat_dim $TANDT_FEAT_DIM \
        -m outputs/tandt/$scene
    
    echo "Completed Tanks and Temples scene: $scene"
done

echo -e "\n=== All TC-GS training tasks completed ==="