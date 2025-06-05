import numpy as np
import os
from pathlib import Path
import math
import open3d
import torch
import csv

def voxel(input_path,target_precision):
    pcd = open3d.io.read_point_cloud(input_path)##########
    plydata = np.asarray(pcd.points)########
    ori_vertex_number=plydata.shape[0]
    print("op:",ori_vertex_number)
    pc_ori_torch = torch.tensor(plydata, dtype=torch.float64)
    pc_ori_torch = pc_ori_torch.cuda()
###########################決定要不要把重複地點拿掉 
    #####這裡會造成損失點數是因為原檔就有重複座標
    pc_o= torch.unique(pc_ori_torch, dim=0)
    pc_o_number=pc_o.shape[0]
    print("point number removing duplicated:",pc_o_number)

    min_values, _ = torch.min(pc_o, dim=0)
    # 找到 x、y、z 軸的最小值中的最小值
    minx,miny,minz=min_values
    ##SHIFT TO POSITIVE DOMAIN####
    pc_o[:, 0] -= minx
    pc_o[:, 1] -= miny
    pc_o[:, 2] -= minz
    # quantization方式直接決定要quantize到哪裡 
    # 先將負數shift到正數domain 
    # 找正數domain x y z 最大值 
    # 最大值≤2**ogap    
    # ogap=ceil(log(maxx)/log(2))
    # scaling factor=2**(target_precision-ogap)  取 round
    # 取round(乘以2**scaling factor)=2**target_precision

    max_values, _ = torch.max(pc_o, dim=0)
    maxx= torch.max(max_values)
    maxx=maxx.cpu()
    ogap=math.ceil(np.log10(maxx)/np.log10(2))
    scaling_factorq = 2 ** (target_precision-ogap)
    pc_ori_torch_q = torch.unique(torch.round(pc_o*scaling_factorq), dim=0)
    pc_q=pc_ori_torch_q.detach().cpu().numpy()
    print("point number after voxel:",pc_q.shape[0])
    return pc_q.astype('int'),minx,miny,minz,ogap,target_precision

def devoxel(plydata,minx,miny,minz,ogap,target_precision,target_path):
    pc_ori_torch = torch.tensor(plydata, dtype=torch.float64)
    pc_ori_torch_q = pc_ori_torch.cuda()
    scaling_factorq = 2 ** (target_precision-ogap)
    pc_dq = torch.unique(pc_ori_torch_q/scaling_factorq, dim=0)
    dq_num=pc_dq.shape[0]
    print("------DEQUANTIZATION DOMAIN------")
    print("point number:",dq_num)
    ##SHIFT TO POINT DOMAIN####
    pc_dq[:, 0] += minx
    pc_dq[:, 1] += miny
    pc_dq[:, 2] += minz
    pc_o_d = pc_dq.detach().cpu().numpy()
    pcdr = open3d.geometry.PointCloud() 
    pcdr.points = open3d.utility.Vector3dVector(pc_o_d)
    new_num_points = len(pcdr.points)

    ######################################
    print(target_path)
    open3d.io.write_point_cloud(target_path, pcdr, write_ascii=True)
    return pc_o_d





