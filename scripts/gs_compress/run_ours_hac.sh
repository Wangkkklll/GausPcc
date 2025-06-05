#!/bin/bash
cd /src/gs_compress/HAC
# MipNeRF360 scenes
for lmbda in 0.004 0.0005; do
  for scene in bicycle garden stump room counter kitchen bonsai flowers treehill; do
    echo "Processing MipNeRF360 scene: $scene with lambda: $lmbda"
    CUDA_VISIBLE_DEVICES=0 python train.py -s /data/mip_nerf360/$scene --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30000 -m outputs/mipnerf360/$scene/$lmbda --lmbda $lmbda
  done
done

# Blending scenes
for lmbda in 0.004 0.0005; do
  for scene in playroom drjohnson; do
    echo "Processing Blending scene: $scene with lambda: $lmbda"
    CUDA_VISIBLE_DEVICES=0 python train.py -s /data/db/$scene --eval --lod 0 --voxel_size 0.005 --update_init_factor 16 --iterations 30000 -m outputs/blending/$scene/$lmbda --lmbda $lmbda
  done
done

# Tanks and Temples scenes
for lmbda in 0.004 0.0005; do
  for scene in truck train; do
    echo "Processing TandT scene: $scene with lambda: $lmbda"
    CUDA_VISIBLE_DEVICES=0 python train.py -s /data/tandt/$scene --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30000 -m outputs/tandt/$scene/$lmbda --lmbda $lmbda
  done
done