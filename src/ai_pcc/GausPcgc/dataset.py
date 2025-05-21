import torch
from torchsparse import SparseTensor
import kit.io as io
from kit.io import kdtree_partition
import random


class PCDataset:
    def __init__(self, file_path_ls, posQ=1, is_pre_quantized=False):
        self.files = io.read_point_clouds(file_path_ls)
        self.posQ = posQ
        self.is_pre_quantized = is_pre_quantized

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        xyz = torch.tensor(self.files[idx], dtype=torch.float)
        feats = torch.ones((xyz.shape[0], 1), dtype=torch.float)

        if not self.is_pre_quantized:
            xyz = xyz / 0.001 
        # xyz = torch.round((xyz + 131072) / self.posQ).int()
        xyz = torch.round((xyz) / self.posQ).int()

        input = SparseTensor(coords=xyz, feats=feats)
        
        return {"input": input}




class PCDataset_Patch:
    def __init__(self, file_path_ls, posQ=1, is_pre_quantized=False):
        self.files = io.read_point_clouds(file_path_ls)
        self.posQ = posQ
        self.is_pre_quantized = is_pre_quantized
        self.max_num = 150000
        # self.max_num = 600000
        # self.max_num = 400000
        # self.max_num = 300000
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # xyz = torch.tensor(self.files[idx], dtype=torch.float)
        xyz = self.files[idx]
        if len(xyz) > self.max_num:
            # print('DBG', len(coords), self.max_num)
            parts = kdtree_partition(xyz, max_num=self.max_num)
            xyz = random.sample(parts, 1)[0]
        xyz = torch.tensor(xyz, dtype=torch.float)
        # xyz = coords
        feats = torch.ones((xyz.shape[0], 1), dtype=torch.float)

        if not self.is_pre_quantized:
            xyz = xyz / 0.001 
        # xyz = torch.round((xyz + 131072) / self.posQ).int()
        xyz = torch.round((xyz) / self.posQ).int()

        input = SparseTensor(coords=xyz, feats=feats)
        
        return {"input": input}