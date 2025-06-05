import os, time
import numpy as np
import torch
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo
from gpcc import gpcc_encode, gpcc_decode
import open3d
device = torch.device('cpu')
from voxelization import voxel,devoxel

class CoordinateCoder():
    """encode/decode coordinates using gpcc
    """
    def __init__(self, filename):
        self.filename = filename
        self.ply_filename = filename + '.ply'##output des
    #voxel&encode
    def encode(self,i,folder_path,target_precision):
        pf=folder_path+i+".ply"
        pq,minx,miny,minz,ogap,target_precision=voxel(pf,target_precision)
        write_ply_ascii_geo(filedir=self.ply_filename, coords=pq)
        gpcc_encode(self.ply_filename, self.filename + '_C.bin')
        os.system('rm '+self.ply_filename)
        return minx,miny,minz,ogap,target_precision
    #decode&devoxel
    def decode(self,minx,miny,minz,ogap,target_precision,target_path):
        gpcc_decode(self.filename + '_C.bin', self.ply_filename)
        coords = read_ply_ascii_geo(self.ply_filename)
        os.system('rm '+self.ply_filename)
        pc_o_dec=devoxel(coords,minx,miny,minz,ogap,target_precision,target_path)
        return pc_o_dec

#filenamel=["mic","hotdog","ficus","materials","ship","chair","lego","drums","room","counter","bonsai","kitchen","stump","bicycle","garden","flowers","treehill","Playroom","DrJohnson"]
if __name__ == '__main__':
    folder_path="/dataset/3dgs/deepblendingply/"
    target_precision=20
    filenamel=["Playroom"]
    for i in filenamel:
        print(i)
        filename="./output/"+i
        with torch.no_grad():
            coordinate_coder = CoordinateCoder(filename)
            # encode
            start_time = time.time()
            minx,miny,minz,ogap,target_precision=coordinate_coder.encode(i,folder_path,target_precision)
            print('Enc Time:\t', round(time.time() - start_time, 3), 's')
            time_enc = round(time.time() - start_time, 3)
            torch.cuda.empty_cache()  # empty cache.
            # decode
            target_path="./dec.ply"
            start_time = time.time()
            xyz_dec=coordinate_coder.decode(minx,miny,minz,ogap,target_precision,target_path)
            print('Dec Time:\t', round(time.time() - start_time, 3), 's')
            time_dec = round(time.time() - start_time, 3)
            torch.cuda.empty_cache()  # empty cache.
        # bitrate
        bytess = np.array([os.path.getsize(filename + postfix) \
                            for postfix in ['_C.bin']])
        #bpps = (bytess*8 / len(x))
        kk=sum(bytess)/(10**6)
        print('bitstream:\t', kk, 'MB')
        print("done")