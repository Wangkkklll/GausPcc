#!/bin/bash

# Training script for HAC-plus Gaussian compression across multiple datasets
# Author: Generated from Python script
# Usage: ./run_all.sh
cd /src/gs_compress/HAC-plus
# MipNeRF360 Dataset
echo "=== Processing MipNeRF360 Dataset ==="
for lmbda in 0.004 0.0005; do  # Optional values: 0.003, 0.002, 0.001, 0.0005
    for scene in bicycle garden stump room counter kitchen bonsai flowers treehill; do
        # Calculate mask_lr_final with proper scaling
        mask_lr_final=$(echo "scale=6; 0.0005 * $lmbda / 0.001" | bc)
        # Ensure mask_lr_final doesn't exceed 0.0015
        if (( $(echo "$mask_lr_final > 0.0015" | bc -l) )); then
            mask_lr_final=0.0015
        fi
        
        echo "Processing scene: $scene, lambda: $lmbda, mask_lr_final: $mask_lr_final"
        CUDA_VISIBLE_DEVICES=1 python train.py \
            -s /data/wkl/gszip_data/mip_nerf360/$scene \
            --eval \
            --lod 0 \
            --voxel_size 0.001 \
            --update_init_factor 16 \
            --iterations 30_000 \
            -m outputs/mipnerf360/$scene/$lmbda \
            --lmbda $lmbda \
            --mask_lr_final $mask_lr_final
    done
done

# Blending Dataset (Deep Blending)
echo "=== Processing Blending Dataset ==="
for lmbda in 0.004 0.0005; do  # Optional values: 0.003, 0.002, 0.001, 0.0005
    for scene in playroom drjohnson; do
        # Calculate mask_lr_final with different scaling factor for blending dataset
        mask_lr_final=$(echo "scale=6; 0.00008 * $lmbda / 0.001" | bc)
        
        echo "Processing scene: $scene, lambda: $lmbda, mask_lr_final: $mask_lr_final"
        CUDA_VISIBLE_DEVICES=1 python train.py \
            -s /data/wkl/gszip_data/db/$scene \
            --eval \
            --lod 0 \
            --voxel_size 0.005 \
            --update_init_factor 16 \
            --iterations 30_000 \
            -m outputs/blending/$scene/$lmbda \
            --lmbda $lmbda \
            --mask_lr_final $mask_lr_final
    done
done

# Tanks and Temples Dataset
echo "=== Processing Tanks and Temples Dataset ==="
for lmbda in 0.004 0.0005; do  # Optional values: 0.003, 0.002, 0.001, 0.0005
    for scene in truck train; do
        # Calculate mask_lr_final with different scaling factor for T&T dataset
        mask_lr_final=$(echo "scale=6; 0.0001 * $lmbda / 0.001" | bc)
        
        echo "Processing scene: $scene, lambda: $lmbda, mask_lr_final: $mask_lr_final"
        CUDA_VISIBLE_DEVICES=1 python train.py \
            -s /data/wkl/gszip_data/tandt/$scene \
            --eval \
            --lod 0 \
            --voxel_size 0.01 \
            --update_init_factor 16 \
            --iterations 30_000 \
            -m outputs/tandt/$scene/$lmbda \
            --lmbda $lmbda \
            --mask_lr_final $mask_lr_final
    done
done

echo "=== All training tasks completed ==="