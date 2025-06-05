import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import torch.nn.init as init
from scene.triplane import TriPlaneField
from scene.plyloader import GaussianPC
from torch.utils.data import DataLoader
from tqdm import tqdm
from scene.rate_distortion import RateDistortionLoss

class Attribute(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=[], args=None, mode=None):
        super(Attribute, self).__init__()
        self.D = D
        self.W = W
        self.resolution = nn.Parameter(torch.tensor(args['kplanes_config']['resolution']), requires_grad=False)
        self.final_dim=args['final_dim']
        self.grid = TriPlaneField(args['bounds'], args['kplanes_config'], args['multires'], args['if_contract'],args['comp_iter'])
        self.net = self.create_net()
        self.args = args

        
    def create_net(self):
        self.feature_out = [nn.Linear(self.grid.feat_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        
        return nn.Sequential(nn.ReLU(),nn.Linear(self.W, self.final_dim))
    
    def query_time(self, rays_pts_emb, scales_emb=None, rotations_emb=None, time_emb=None, itr=-1):
        grid_feature, rate_y = self.grid(rays_pts_emb, itr=itr)
        h = grid_feature
        h = self.feature_out(h)
        return h, rate_y

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None, time_emb=None, itr=-1):
        return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb, itr=itr)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic(self,rays_pts_emb, scales_emb=None, rotations_emb=None, opacity_emb=None, time_emb=None, itr=-1):
        hidden, rate_y = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_emb, itr=itr)
        result = self.net(hidden)        
        return result, rate_y

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                # print('a', name)
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        return list(self.grid.grids.parameters())

    def get_arm_parameters(self):
        return list(self.grid.arm.parameters()) 
    def get_arm2_parameters(self):
        return list(self.grid.arm2.parameters()) 
    def get_arm3_parameters(self):
        return list(self.grid.arm3.parameters()) 

class attribute_network(nn.Module):
    def __init__(self, args_dict):
        super(attribute_network, self).__init__()
        self.attribute_net = Attribute(W=args_dict['net_width'], D=args_dict['net_depth'], args=args_dict)
        self.apply(initialize_weights)

    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None, itr=-1):
        return self.forward_dynamic(point, itr=itr)

    def forward_static(self, points):
        points = self.attribute_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None, itr=-1):
        result, rate_y = self.attribute_net(point, itr=itr)
        return result, rate_y
    def get_mlp_parameters(self):
        return self.attribute_net.get_mlp_parameters()
    def get_grid_parameters(self):
        return self.attribute_net.get_grid_parameters()
    def get_arm_parameters(self):
        return self.attribute_net.get_arm_parameters()
    def get_arm2_parameters(self):
        return self.attribute_net.get_arm2_parameters()
    def get_arm3_parameters(self):
        return self.attribute_net.get_arm3_parameters()
   

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)