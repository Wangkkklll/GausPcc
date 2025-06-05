import os

for lmbda in [0.004,0.0005]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai', 'flowers', 'treehill']):
        mask_lr_final = 0.0005 * lmbda / 0.001
        mask_lr_final = min(mask_lr_final, 0.0015)
        one_cmd = f'CUDA_VISIBLE_DEVICES={1} python train.py -s /data/wkl/gszip_data/mip_nerf360/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -m outputs/mipnerf360/{scene}/{lmbda} --lmbda {lmbda} --mask_lr_final {mask_lr_final}'
        os.system(one_cmd)


for lmbda in [0.004,0.0005]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['playroom', 'drjohnson']):
        mask_lr_final = 0.00008 * lmbda / 0.001
        one_cmd = f'CUDA_VISIBLE_DEVICES={1} python train.py -s /data/wkl/gszip_data/db/{scene} --eval --lod 0 --voxel_size 0.005 --update_init_factor 16 --iterations 30_000 -m outputs/blending/{scene}/{lmbda} --lmbda {lmbda} --mask_lr_final {mask_lr_final}'
        os.system(one_cmd)



for lmbda in [0.004,0.0005]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['truck', 'train']):
        mask_lr_final = 0.0001 * lmbda / 0.001
        one_cmd = f'CUDA_VISIBLE_DEVICES={1} python train.py -s /data/wkl/gszip_data/tandt/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt/{scene}/{lmbda} --lmbda {lmbda} --mask_lr_final {mask_lr_final}'
        os.system(one_cmd)
