import os
from argparse import ArgumentParser, Namespace
import sys
import glob

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("-s", "--scene", type=str, default=None, help="Scene to train on (leave empty to process all)")
args = parser.parse_args(sys.argv[1:])

# 定义所有数据集配置
datasets = {
    "mip_nerf360": {
        "s_pth": "/data/wkl/gszip_data/mip_nerf360",
        "project_prefix": "CAT3DGS_mipsnerf360",
        "txt_prefix": "output_result/mipsnerf360",
        "m_pth": "output_log/mipsnerf360",
        "voxel_size": 0.001,
        "cmd_extra": "--chcm_slices_list 5 10 15 20"
    },
    "db": {
        "s_pth": "/data/wkl/gszip_data/db",
        "project_prefix": "CAT3DGS_db",
        "txt_prefix": "output_result/db",
        "m_pth": "output_log/db",
        "voxel_size": 0.005,
        "cmd_extra": "--chcm_for_offsets --chcm_for_scaling"
    },
    "tandt": {
        "s_pth": "/data/wkl/gszip_data/tandt",
        "project_prefix": "CAT3DGS_tnt",
        "txt_prefix": "output_result/tnt",
        "m_pth": "output_log/tnt",
        "voxel_size": 0.01,
        "cmd_extra": "--chcm_slices_list 5 10 15 20"
    }
}

# 依次处理所有数据集
for dataset_name, config in datasets.items():
    print(f"\n\n=== 开始处理数据集: {dataset_name} ===\n")
    
    s_pth = config["s_pth"]
    project_prefix = config["project_prefix"]
    txt_prefix = config["txt_prefix"]
    m_pth = config["m_pth"]
    voxel_size = config["voxel_size"]
    cmd_extra = config["cmd_extra"]
    
    # 获取场景列表
    if args.scene:
        scenes = [args.scene]
    else:
        scenes = [os.path.basename(scene_path) for scene_path in glob.glob(f"{s_pth}/*") 
                if os.path.isdir(scene_path)]
    
    print(f"将处理数据集 {dataset_name} 中的以下场景: {scenes}")
    
    # 循环处理所有场景
    for scene in scenes:
        print(f"\n=== 处理场景: {scene} ===")
        
        # for lmbda in [0.04, 0.002]:  # [0.002, 0.004, 0.01, 0.015, 0.03, 0.04]
        # for lmbda in [0.04,0.002,0.004, 0.01]: 
        for lmbda in [0.001]:
            project_name = f"{project_prefix}_{scene}"
            txt_name = f"{txt_prefix}/{scene}"
            
            one_cmd = f'CUDA_VISIBLE_DEVICES=1 python train.py --eval --lod 0 --voxel_size {voxel_size} --update_init_factor 16 --iterations 40_000 --test_iterations 40000 -s {s_pth}/{scene} -m {m_pth}/{scene}/{lmbda} --exp_name {scene}_{lmbda} --lmbda {lmbda} --project_name {project_name} --cam_mask 1 --txt_name {txt_name} {cmd_extra}'
            
            print(f"执行命令: {one_cmd}")
            os.system(one_cmd)