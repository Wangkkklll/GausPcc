import os

# 基础参数
gpu_id = 6
iterations = 30000

# mipnerf360 数据集参数
mipnerf360_params = {
    'voxel_size': 0.001,
    'update_init_factor': 16,
    'appearance_dim': 0,
    'ratio': 1,
    'tri_resolution': 32,
    'feat_dim': 50
}

# db 数据集参数
db_params = {
    'voxel_size': 0.005,
    'update_init_factor': 16,
    'appearance_dim': 0,
    'ratio': 1,
    'tri_resolution': 16,
    'feat_dim': 50
}

# tandt 数据集参数
tandt_params = {
    'voxel_size': 0.01,
    'update_init_factor': 16,
    'appearance_dim': 0,
    'ratio': 1,
    'tri_resolution': 16,
    'feat_dim': 50
}

# 1. mipnerf360 数据集
mipnerf360_scenes = [
    'bicycle', 'garden', 'stump', 'room', 'counter', 'kitchen', 'bonsai', 'flowers', 'treehill'
]
for scene in mipnerf360_scenes:
    cmd = (
        f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py '
        f'--eval -s /public/DATA/wkl/3d_edit_dataset/mip_nerf360/{scene} '
        f'--lod 0 --gpu -1 --voxel_size {mipnerf360_params["voxel_size"]} '
        f'--update_init_factor {mipnerf360_params["update_init_factor"]} '
        f'--appearance_dim {mipnerf360_params["appearance_dim"]} '
        f'--ratio {mipnerf360_params["ratio"]} '
        f'--iterations {iterations} '
        f'--tri_resolution {mipnerf360_params["tri_resolution"]} '
        f'--feat_dim {mipnerf360_params["feat_dim"]} '
        f'-m outputs/mipnerf360/{scene}'
    )
    print(cmd)
    os.system(cmd)

# 2. db 数据集
db_scenes = ['playroom', 'drjohnson']
for scene in db_scenes:
    cmd = (
        f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py '
        f'--eval -s /public/DATA/wkl/3d_edit_dataset/db/{scene} '
        f'--lod 0 --gpu -1 --voxel_size {db_params["voxel_size"]} '
        f'--update_init_factor {db_params["update_init_factor"]} '
        f'--appearance_dim {db_params["appearance_dim"]} '
        f'--ratio {db_params["ratio"]} '
        f'--iterations {iterations} '
        f'--tri_resolution {db_params["tri_resolution"]} '
        f'--feat_dim {db_params["feat_dim"]} '
        f'-m outputs/blending/{scene}'
    )
    print(cmd)
    os.system(cmd)

# 3. tandt 数据集
tandt_scenes = ['truck', 'train']
for scene in tandt_scenes:
    cmd = (
        f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py '
        f'--eval -s /public/DATA/wkl/3d_edit_dataset/tandt/{scene} '
        f'--lod 0 --gpu -1 --voxel_size {tandt_params["voxel_size"]} '
        f'--update_init_factor {tandt_params["update_init_factor"]} '
        f'--appearance_dim {tandt_params["appearance_dim"]} '
        f'--ratio {tandt_params["ratio"]} '
        f'--iterations {iterations} '
        f'--tri_resolution {tandt_params["tri_resolution"]} '
        f'--feat_dim {tandt_params["feat_dim"]} '
        f'-m outputs/tandt/{scene}'
    )
    print(cmd)
    os.system(cmd)