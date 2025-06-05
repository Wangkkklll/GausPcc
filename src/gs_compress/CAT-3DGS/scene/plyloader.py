import os, subprocess
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData
from torch.utils.data import DataLoader
from torchvision import transforms
from random import random
from plyfile import PlyData, PlyElement
from torch import nn
import numpy as np
from simple_knn._C import distCUDA2

class GaussianPC(torchData):
    def __init__(self, xyz, attribute, mode):
        super().__init__()
        self._xyz = xyz.detach().cpu().numpy().astype(np.float32)
        
        if mode == 'o':
            s1, s2, s3 = attribute.shape
            temp = torch.reshape(attribute, (attribute.shape[0], 30))
            self._attribute = temp.detach().cpu().numpy().astype(np.float32)
            
        else:
            self._attribute = attribute.detach().cpu().numpy().astype(np.float32)
        # if mode == 'o':
        #     self._attribute = np.minimum(self._attribute, 7) / 7
        # if mode == 's':
        #     self._attribute = np.exp(self._attribute)
        
        
        
        
        
        
    def get_covariance_matrix(self):
        cov = self.covariance_matrix(self._attribute)
        
        
        return cov
        
    def covariance_matrix(self, data):
        data = torch.Tensor(data) 
        mean_vector = torch.mean(data, dim=0, keepdim=True)
        
        diff = data - mean_vector
        
        cov_matrix = torch.matmul(diff.t(), diff) / (data.size(0) - 1)
        
        return cov_matrix
        
    def __len__(self):
        return self._xyz.shape[0]
    def __getitem__(self, id):
        return self._xyz[id], self._attribute[id]

    
    
if __name__ == '__main__':
    a = GaussianPC('/home/pc3435/Kelvin/Compact-3DGS/output/qq14/point_cloud/iteration_30000/point_cloud.ply')
    print(a[3])
        