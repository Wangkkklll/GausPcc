import os
import multiprocessing

import numpy as np
from tqdm import tqdm

import open3d as o3d

#######################################
#######################################

def read_points(filedir):
    if os.path.splitext(filedir)[-1] == '.bin':
        return np.fromfile(filedir, dtype=np.float32).reshape(-1, 4)[:, :3]
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:, 0:3]
    return coords

def read_point_clouds(file_path_list):
    print('Loading point clouds...')
    with multiprocessing.Pool(64) as p:
        pcs = list(tqdm(p.imap(read_points, file_path_list), total=len(file_path_list)))
    return pcs

def save_ply_ascii_geo(coords, filedir):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype("float32"))
    if os.path.exists(filedir):
        os.system("rm " + filedir)
    f = open(filedir, "a+")
    f.writelines(["ply\n", "format ascii 1.0\n"])
    f.write("element vertex " + str(coords.shape[0]) + "\n")
    f.writelines(["property float x\n", "property float y\n", "property float z\n"])
    f.write("end_header\n")
    coords = coords.astype("float32")
    for xyz in coords:
        f.writelines([str(xyz[0]), " ", str(xyz[1]), " ", str(xyz[2]), "\n"])
    f.close()




def kdtree_partition(points, max_num, n_parts=None):
    parts = []
    if n_parts is not None: max_num = len(points)/n_parts + 2
    class KD_node:  
        def __init__(self, point=None, LL = None, RR = None):  
            self.point = point  
            self.left = LL  
            self.right = RR
    def createKDTree(root, data):
        if len(data) <= max_num:
            parts.append(data)
            return
        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]

        point = data_sorted[int(len(data)/2)]
        root = KD_node(point)
        root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))])  
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):]) 
        return root
    init_root = KD_node(None)
    root = createKDTree(init_root, points)

    return parts

#######################################
#######################################

