#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch_scatter import scatter_max
from sklearn.neighbors import LocalOutlierFactor
from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized
from scene.attribute import attribute_network

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    anchor_round_digits, Q_anchor, \
    encoder_anchor, decoder_anchor, \
    encoder, decoder, \
    encoder_gaussian, decoder_gaussian, \
    get_binary_vxl_size

from scene.triplane import *
import time
import subprocess
from torch import Tensor
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from scene.bitstream.encode import encode, RangeCoder, get_ac_max_val_latent, write_header
from scene.bitstream.decode import decode, compute_offset, fast_get_neighbor
import os
from scene.arm import ArmMLP, get_mu_scale, get_neighbor
import numpy as np
from utils.pcc_utils import calculate_morton_order,compress_point_cloud,decompress_point_cloud

bit2MB_scale = 8 * 1024 * 1024
Q_EXP_SCALE = 2**4


class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int=32,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank = False,
                 n_features_per_level: int=2,
                 chcm_slices_list = [25, 25],
                 chcm_for_offsets = False,
                 chcm_for_scaling = False,
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False,
                 attribute_config: dict=None
                 ):
        super().__init__()
        
        feat_dim = 50
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        print("self.update_init_factor",self.update_init_factor)
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version
        
        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._mask = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.temp_coder_list = None
        self.setup_functions()

        

        

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
            # nn.Linear(feat_dim, 7),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        

        self.mlp_context_from_f1 = nn.Sequential(
            nn.Linear(feat_dim//2, 2*feat_dim),
            nn.ReLU(True),
            nn.Linear(2*feat_dim, feat_dim),
        ).cuda()

        self.mlp_chcm_list = nn.ModuleList()
        self.num_feat_slices = len(chcm_slices_list)
        self.chcm_slices_list = chcm_slices_list
        input_dim = 0
        for i in range(self.num_feat_slices-1):
            input_dim += chcm_slices_list[i]
            self.mlp_chcm_list.append(nn.Sequential(
                nn.Linear(input_dim, 2*feat_dim),
                nn.ReLU(True),
                nn.Linear(2*feat_dim, 2*chcm_slices_list[i+1]),
            ).cuda())

        self.chcm_for_offsets = chcm_for_offsets
        if chcm_for_offsets:
            self.mlp_chcm_offsets = nn.Sequential(
                nn.Linear(self.feat_dim, 2*feat_dim),
                nn.ReLU(True),
                nn.Linear(2*feat_dim, 6*self.n_offsets),
            ).cuda()
        self.chcm_for_scaling = chcm_for_scaling
        if chcm_for_scaling:
            self.mlp_chcm_scaling = nn.Sequential(
                nn.Linear(self.feat_dim, 2*feat_dim),
                nn.ReLU(True),
                nn.Linear(2*feat_dim, 12),
            ).cuda()

        #self.feature_net = attribute_network(attribute_config, 'sof').cuda()
        self.attribute_config=attribute_config
        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()

    def get_mlp_size(self, digit=32):
        # mlp_size = 0
        # for n, p in self.named_parameters():
        #     if 'mlp' in n and 'deform' not in n:
        #         mlp_size += p.numel()*digit
        pmlp_o=self.count_parameters(self.mlp_opacity)
        pmlp_rs=self.count_parameters(self.mlp_cov)
        pmlp_color=self.count_parameters(self.mlp_color)
        parm=self.count_parameters(self.feature_net.attribute_net.grid.arm)
        parm2=self.count_parameters(self.feature_net.attribute_net.grid.arm2)
        parm3=self.count_parameters(self.feature_net.attribute_net.grid.arm3)
        pmlp_fa=sum(p.numel() for p in self.feature_net.get_mlp_parameters())
        pmlp_chcm = 0
        for i in range(self.num_feat_slices-1):
            pmlp_chcm+=self.count_parameters(self.mlp_chcm_list[i])
        if self.chcm_for_offsets:
            pmlp_chcm+=self.count_parameters(self.mlp_chcm_offsets)
        if self.chcm_for_scaling:
            pmlp_chcm+=self.count_parameters(self.mlp_chcm_scaling)
        total_mlp_p=pmlp_o+pmlp_rs+pmlp_color+parm+parm2+parm3+pmlp_fa+pmlp_chcm
        print("# param of all MLP:",f"{total_mlp_p/10**6}M")
        print("Size of all MLP(32bit):",f"{total_mlp_p*4/10**6}MB")  
        mlp_size = total_mlp_p*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.mlp_chcm_list.eval()
        if self.chcm_for_offsets:
            self.mlp_chcm_offsets.eval()
        if self.chcm_for_scaling:
            self.mlp_chcm_scaling.eval()

        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.chcm_for_offsets:
            self.mlp_chcm_offsets.train()
        if self.chcm_for_scaling:
            self.mlp_chcm_scaling.train()
        self.mlp_chcm_list.train()

        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self.max_radii2D,
            self.optimizer.state_dict(),
            self.feature_grid_optimizer.state_dict(),
            self.feature_net_optimizer.state_dict(),
            self.feature_arm_optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
        self.max_radii2D,
        opt_dict,
        feature_grid_opt_dict,
        feature_net_opt_dict,
        feature_arm_opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.training_setup_triplane(training_args)
        print(f"optimizer state_dict: {opt_dict}")
        self.optimizer.load_state_dict(opt_dict)
        self.feature_grid_optimizer.load_state_dict(feature_grid_opt_dict)
        self.feature_net_optimizer.load_state_dict(feature_net_opt_dict)
        self.feature_arm_optimizer.load_state_dict(feature_arm_opt_dict)
        torch.cuda.empty_cache()

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0*self.scaling_activation(self._scaling)

    def get_mask(self, weighted_mask=None):
        if self.decoded_version:
            return self._mask
        mask_sig = torch.sigmoid(self._mask)
        if weighted_mask is not None:
            # print(mask_sig.shape, weighted_mask.shape)
            mask_sig = mask_sig * weighted_mask
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    
    def get_mask_anchor(self, weighted_mask=None):
        with torch.no_grad():
            if self.decoded_version:
                mask_anchor = (torch.sum(self._mask, dim=1)[:, 0]) > 0
                return mask_anchor
            mask_sig = torch.sigmoid(self._mask)
            if weighted_mask is not None:
                mask_sig = mask_sig * weighted_mask
            mask = ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig
            mask_anchor = (torch.sum(mask, dim=1)[:, 0]) > 0
            return mask_anchor

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_context_from_f1(self):
        return self.mlp_context_from_f1
    
    @property
    def get_chcm_mlp_list(self):
        return self.mlp_chcm_list
    
    @property
    def get_chcm_mlp_offsets(self):
        if self.chcm_for_offsets:
            return self.mlp_chcm_offsets
        else:
            return None
    
    @property
    def get_chcm_mlp_scaling(self):
        if self.chcm_for_scaling:
            return self.mlp_chcm_scaling
        else:
            return None

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor = torch.round(self._anchor / self.voxel_size) * self.voxel_size
        anchor = anchor.detach() + (self._anchor - self._anchor.detach())
        return anchor

    @torch.no_grad()
    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        print('anchor_bound_updated')

    @torch.no_grad()
    def set_feature_net(self, w=None):
        _anchor = self.get_anchor
        anchornp=_anchor.detach().cpu().numpy()
        
        x = anchornp[:,0]
        y = anchornp[:,1]
        z = anchornp[:,2]
        xyz = np.vstack([x, y, z]).T
        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.05)
        labels = lof.fit_predict(xyz)

        dense_points = torch.tensor(xyz[labels == 1], dtype=torch.float32).cuda()

        mean = torch.mean(dense_points, dim=0)
        centered_points = dense_points - mean
        cov_matrix  = torch.cov(centered_points.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        rotated_points = torch.matmul(centered_points, sorted_eigenvectors)

        standardized_points = rotated_points / torch.sqrt(eigenvalues[sorted_indices])
        
        print(f"mean: {mean}")
        print(f"eigenvalues: {eigenvalues[sorted_indices]}")
        print(f"eigenvectors: {sorted_eigenvectors}")
        w=None
        anchor_num=self.get_anchor.shape[0]
        x=round((anchor_num / 36) ** 0.5)
        self.attribute_config['kplanes_config']['resolution'] = [x,x,x]
        self.feature_net = attribute_network(self.attribute_config).cuda()
        self.feature_net.attribute_net.grid.set_aabb(standardized_points, self.x_bound_max, self.x_bound_min)
        self.feature_net.attribute_net.grid.set_rotation_matrix(sorted_eigenvectors, mean, torch.sqrt(eigenvalues[sorted_indices]))

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets, 1)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},


            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

            ]

        self.l_param=[]
        for item in l:
            if isinstance(item['params'], list):
                for param in item['params']:
                    self.l_param.append(param)
            else:
                self.l_param.append(param) 
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.mask_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.mask_lr_delay_mult,
                                                    max_steps=training_args.mask_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)

        

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                    lr_final=training_args.mlp_deform_lr_final,
                                                    lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                    max_steps=training_args.mlp_deform_lr_max_steps)

    def training_setup_triplane(self, training_args):
        feature_grid_l = [
            {'params': self.feature_net.get_grid_parameters(), 'lr': training_args.feature_lr, "name": "mlp_color"},
        ]
        self.feature_grid_optimizer = torch.optim.Adam(feature_grid_l, lr=0.0, eps=1e-15)
        
        feature_net_l = [
            {'params': self.feature_net.get_mlp_parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            {'params': self.mlp_chcm_list.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        ]
        if self.chcm_for_offsets:
            feature_net_l.append({'params': self.mlp_chcm_offsets.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"})
        if self.chcm_for_scaling:
            feature_net_l.append({'params': self.mlp_chcm_scaling.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"})
        self.feature_net_optimizer = torch.optim.Adam(feature_net_l, lr=0.0, eps=1e-15)
        
        feature_arm_l = [
            {'params': self.feature_net.get_arm_parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_color"},
            {'params': self.feature_net.get_arm2_parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_color"},
            {'params': self.feature_net.get_arm3_parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_color"},
        ]
        self.feature_arm_optimizer = torch.optim.Adam(feature_arm_l, lr=0.0, eps=1e-15)    
        
        
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters())

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mask":
                lr = self.mask_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr


        if  iteration > 10000:
            for param_group in self.feature_net_optimizer.param_groups:
                if param_group["name"] == "mlp_color" and iteration > 10000:
                        lr = self.mlp_color_scheduler_args(iteration-10000)
                        param_group['lr'] = lr

            for param_group in self.feature_grid_optimizer.param_groups:
                if param_group["name"] == "mlp_color" and iteration > 10000:
                    lr = self.mlp_color_scheduler_args(iteration-10000)
                    param_group['lr'] = lr        

            for param_group in self.feature_arm_optimizer.param_groups:
                if param_group["name"] == "mlp_color" and iteration > 15000:
                    lr = self.mlp_color_scheduler_args(iteration-15000)
                    param_group['lr'] = lr
                

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1]*self._mask.shape[2]):
            l.append('f_mask_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mask = self._mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        N = anchor.shape[0]
        opacities = opacities[:N]
        rotation = rotation[:N]

        attributes = np.concatenate((anchor, normals, offset, mask, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key = lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        del plydata
        torch.cuda.empty_cache()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self,path):
        mkdir_p(os.path.dirname(path))

        if self.use_feat_bank:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
            }, path)
        else:
            print('saving mlp checkpoints')
            # for name, param in self.feature_net.named_parameters():
            #     print(name, param.shape)
            #     print(param)
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'context_mlp': self.mlp_chcm_list.state_dict(),
                'context_mlp_offsets': self.mlp_chcm_offsets.state_dict() if self.chcm_for_offsets else None,
                'context_mlp_scaling': self.mlp_chcm_scaling.state_dict() if self.chcm_for_scaling else None,
                'feat_base': self.feature_net.state_dict(),
            }, path)


    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        self.mlp_chcm_list.load_state_dict(checkpoint['context_mlp'])
        if self.chcm_for_offsets:
            self.mlp_chcm_offsets.load_state_dict(checkpoint['context_mlp_offsets'])
        if self.chcm_for_scaling:
            self.mlp_chcm_scaling.load_state_dict(checkpoint['context_mlp_scaling'])
        
        tri_resolution = checkpoint['feat_base']['attribute_net.resolution']
        print(f'triplance resolution: {tri_resolution}')
        self.attribute_config['kplanes_config']['resolution'] = tri_resolution
        self.feature_net = attribute_network(self.attribute_config).cuda()

        self.feature_net.load_state_dict(checkpoint['feat_base'])
        # print(f'feat_base: {self.feature_net}')
        print('saving mlp checkpoints')
        # for name, param in self.feature_net.named_parameters():
        #     print(name, param.shape)
        #     print(param)
        torch.cuda.empty_cache()

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    @torch.no_grad()
    def estimate_final_bits(self, weighted_mask):
        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2
        mask_anchor = self.get_mask_anchor(weighted_mask)
        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        # print(f'1, {torch.isnan(_feat).sum()}')
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask(weighted_mask)[mask_anchor]
        feat_context, feat_rate = self.feature_net(_anchor, itr=-1)
        scales, means, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(feat_context, split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
        list_of_means = list(torch.split(means, split_size_or_sections=self.chcm_slices_list, dim=-1))
        list_of_scales = list(torch.split(scales, split_size_or_sections=self.chcm_slices_list, dim=-1))
        

        
        Q_feat = Q_feat * (1.1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))
        _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        # print(f'2, {torch.isnan(_feat).sum()}')
        feat_list = list(torch.split(_feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
        # print(f'3, {torch.isnan(feat1).sum()}, {torch.isnan(feat2).sum()}')
        bit_feat_list = []
        for i in range(self.num_feat_slices):
            if i == 0:
                bit_feat = self.entropy_gaussian.forward(feat_list[i], list_of_means[i], list_of_scales[i], Q_feat)
                bit_feat_list.append(torch.sum(bit_feat).item())
                decoded_feat = feat_list[0]
            else:
                dmean, dscale = torch.split(self.mlp_chcm_list[i-1](decoded_feat), split_size_or_sections=[self.chcm_slices_list[i], self.chcm_slices_list[i]], dim=-1)
                mean = list_of_means[i] + dmean
                scale = list_of_scales[i] + dscale
                bit_feat = self.entropy_gaussian.forward(feat_list[i], mean, scale, Q_feat)
                bit_feat_list.append(torch.sum(bit_feat).item())
                decoded_feat = torch.cat([decoded_feat, feat_list[i]], dim=-1)


        # dmean2, dscale2 = torch.split(self.mlp_context_from_f1(feat1), split_size_or_sections=[self.feat_dim//2, self.feat_dim//2], dim=-1)
        # scale2 = scale2 + dscale2
        # mean2 = mean2 + dmean2
        # mean = torch.cat([mean1, mean2], dim=-1)
        # scale = torch.cat([scale1, scale2], dim=-1)
        if self.chcm_for_scaling:
            dmean_scaling, dscale_scaling = torch.split(self.mlp_chcm_scaling(decoded_feat), split_size_or_sections=[6, 6], dim=-1)
            mean_scaling = mean_scaling + dmean_scaling
            scale_scaling = scale_scaling + dscale_scaling
        if self.chcm_for_offsets:
            dmean_offsets, dscale_offsets = torch.split(self.mlp_chcm_offsets(decoded_feat), split_size_or_sections=[3*self.n_offsets, 3*self.n_offsets], dim=-1)
            mean_offsets = mean_offsets + dmean_offsets
            scale_offsets = scale_offsets + dscale_offsets
        grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()
        offsets = offsets.view(-1, 3*self.n_offsets)
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets)
        # print(f'4, {torch.isnan(feat1).sum()}, {torch.isnan(feat2).sum()}')
        # bit_feat1 = self.entropy_gaussian.forward(feat1, mean1, scale1, Q_feat)
        # bit_feat2 = self.entropy_gaussian.forward(feat2, mean2, scale2, Q_feat)
        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
        bit_offsets = bit_offsets * mask_tmp
        bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        # bit_feat1 = torch.sum(bit_feat1).item()
        # bit_feat2 = torch.sum(bit_feat2).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        
        bit_masks = get_binary_vxl_size(_mask)[1].item()
        ftrirate=feat_rate / bit2MB_scale
        log_info = f"\nEstimated sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"total feat {round(sum(bit_feat_list)/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Triplane_f {round(ftrirate.item(),4)}," \
                   f"Total {round((bit_anchor + sum(bit_feat_list) + bit_scaling + bit_offsets + bit_masks+ self.get_mlp_size()[0])/bit2MB_scale+ftrirate.item(), 4)}"
        return log_info

    @torch.no_grad()
    def conduct_encoding(self, pre_path_name, weighted_mask):

        t_codec = 0

        torch.cuda.synchronize(); t1 = time.time()
        print('Start encoding ...')

        mask_anchor = self.get_mask_anchor(weighted_mask)

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask(weighted_mask)[mask_anchor]


        _anchor_int = torch.round(_anchor / self.voxel_size)
        sorted_indices = calculate_morton_order(_anchor_int)
        _anchor_int = _anchor_int[sorted_indices]
        npz_path= os.path.join(pre_path_name, 'xyz_pcc.bin')
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(parent_dir, 'GausPcgc/best_model_ue_4stage_conv.pt')  #   gauspcgc model path
        out = compress_point_cloud(_anchor_int,model_path, npz_path)
        bits_xyz = out['file_size_bits']

        _anchor = _anchor_int * self.voxel_size
        _feat = _feat[sorted_indices]
        _grid_offsets = _grid_offsets[sorted_indices]
        _scaling = _scaling[sorted_indices]
        _mask = _mask[sorted_indices]

        N = _anchor.shape[0]






        MAX_batch_size = 500
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_lists = [[] for _ in range(self.num_feat_slices)]
        
        # bit_feat2_list = []
        bit_scaling_list = []
        bit_offsets_list = []
        anchor_infos_list = []
        indices_list = []
        min_feat_lists = [[] for _ in range(self.num_feat_slices)]
        max_feat_lists = [[] for _ in range(self.num_feat_slices)]
        
        min_scaling_list = []
        max_scaling_list = []
        min_offsets_list = []
        max_offsets_list = []

        feat_lists = [[] for _ in range(self.num_feat_slices)]
        scaling_list = []
        offsets_list = []

        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        # torch.save(_anchor, os.path.join(pre_path_name, 'anchor.pkl'))

        for s in range(steps):
            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            feat_b_name_list = []
            for i in range(self.num_feat_slices):
                feat_b_name_list.append(os.path.join(pre_path_name, f'feat{i}.b').replace('.b', f'_{s}.b'))
            # feat2_b_name = os.path.join(pre_path_name, 'feat2.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            indices = torch.tensor(data=range(N_num), device='cuda', dtype=torch.long)  # [N_num]
            anchor_infos = None
            anchor_infos_list.append(anchor_infos)
            indices_list.append(indices+N_start)

            anchor_sort = _anchor[N_start:N_end][indices]  # [N_num, 3]
            feat_context, feat_rate = self.feature_net(anchor_sort, itr=-1)
            scales, means, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(feat_context, split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
            means = list(torch.split(means, split_size_or_sections=self.chcm_slices_list, dim=-1))
            scales = list(torch.split(scales, split_size_or_sections=self.chcm_slices_list, dim=-1))
        
        
            Q_feat_list = [Q_feat * (1.1 + torch.tanh(Q_feat_adj.contiguous().repeat(1, ch).view(-1))) for ch in self.chcm_slices_list]
            # Q_feat_adj2 = Q_feat_adj.contiguous().repeat(1, self.feat_dim//2)
            

            feat = _feat[N_start:N_end][indices]  # [N_num*32]

            _feat_list = list(torch.split(feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
            
            for i in range(self.num_feat_slices):
                if i == 0:
                    # print(feat[0])
                    temp_feat = _feat_list[i].reshape(-1)
                    temp_feat = STE_multistep.apply(temp_feat, Q_feat_list[i], _feat.mean())
                    
                    
                    mean = means[i].reshape(-1)
                    scale = torch.clamp(scales[i].reshape(-1), min=1e-9)
                    # print(f'feat shape: {feat.shape}, mean shape: {mean.shape}, scale shape: {scale.shape}, Q_feat shape: {Q_feat_list[i].shape}')
                    bit_feat, min_feat, max_feat = encoder_gaussian(temp_feat, mean, scale, Q_feat_list[i], file_name=feat_b_name_list[i])
                    # bit_feat = bit_feat.reshape(self._anchor.shape[0], -1)
                    bit_feat_lists[i].append(bit_feat)
                    min_feat_lists[i].append(min_feat)
                    max_feat_lists[i].append(max_feat)
                    temp_feat = temp_feat.reshape(N_num, -1)
                    # print(f'reshape {f1-feat}')
                    feat_lists[i].append(temp_feat)
                    decoded_feat = temp_feat
                    
                else:
                    temp_feat = _feat_list[i].reshape(-1)
                    temp_feat = STE_multistep.apply(temp_feat, Q_feat_list[i], _feat.mean())
                    
                    dmean, dscale = torch.split(self.mlp_chcm_list[i-1](decoded_feat), split_size_or_sections=[self.chcm_slices_list[i], self.chcm_slices_list[i]], dim=-1)
                    mean = (means[i] + dmean).reshape(-1)
                    scale = torch.clamp((scales[i] + dscale).reshape(-1), min=1e-9)
                    # if s == 5:
                    #     print('aaaaaaa')
                    #     print(mean[0], scale[0])
                    bit_feat, min_feat, max_feat = encoder_gaussian(temp_feat, mean, scale, Q_feat_list[i], file_name=feat_b_name_list[i])
                    bit_feat_lists[i].append(bit_feat)
                    min_feat_lists[i].append(min_feat)
                    max_feat_lists[i].append(max_feat)
                    # bit_feat = bit_feat.reshape(self._anchor.shape[0], -1)
                    temp_feat = temp_feat.reshape(N_num, -1)
                    feat_lists[i].append(temp_feat)
                    decoded_feat = torch.cat([decoded_feat, temp_feat], dim=-1)





            # feat1, feat2 = torch.split(feat, split_size_or_sections=[self.feat_dim//2, self.feat_dim//2], dim=-1)
            # feat1 = STE_multistep.apply(feat1, Q_feat1, _feat.mean())
            # feat2 = STE_multistep.apply(feat2, Q_feat2, _feat.mean())
            # feat2 = feat2.reshape(-1)
            # Q_feat1 = Q_feat1.reshape(-1)
            # Q_feat2 = Q_feat2.reshape(-1)
            # dmean2, dscale2 = torch.split(self.mlp_context_from_f1(feat1), split_size_or_sections=[self.feat_dim//2, self.feat_dim//2], dim=-1)
            # feat1 = feat1.reshape(-1)
            # scale2 = scale2 + dscale2
            # mean2 = mean2 + dmean2
            # mean1 = mean1.reshape(-1)
            # mean2 = mean2.reshape(-1)
            # scale1 = torch.clamp(scale1.reshape(-1), min=1e-9)
            # scale2 = torch.clamp(scale2.reshape(-1), min=1e-9)
            # torch.cuda.synchronize(); t0 = time.time()
            # bit_feat1, min_feat1, max_feat1 = encoder_gaussian(feat1, mean1, scale1, Q_feat1, file_name=feat1_b_name)
            # bit_feat2, min_feat2, max_feat2 = encoder_gaussian(feat2, mean2, scale2, Q_feat2, file_name=feat2_b_name)
            # torch.cuda.synchronize(); t_codec += time.time() - t0
            # bit_feat1_list.append(bit_feat1)
            # min_feat1_list.append(min_feat1)
            # max_feat1_list.append(max_feat1)
            # bit_feat2_list.append(bit_feat2)
            # min_feat2_list.append(min_feat2)
            # max_feat2_list.append(max_feat2)

            # feat1_list.append(feat1)
            # feat2_list.append(feat2)

            if self.chcm_for_scaling:
                dmean_scaling, dscale_scaling = torch.split(self.mlp_chcm_scaling(decoded_feat), split_size_or_sections=[6, 6], dim=-1)
                mean_scaling = mean_scaling + dmean_scaling
                scale_scaling = scale_scaling + dscale_scaling
            if self.chcm_for_offsets:
                dmean_offsets, dscale_offsets = torch.split(self.mlp_chcm_offsets(decoded_feat), split_size_or_sections=[3*self.n_offsets, 3*self.n_offsets], dim=-1)
                mean_offsets = mean_offsets + dmean_offsets
                scale_offsets = scale_offsets + dscale_offsets

            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            # Q_feat1 = Q_feat * (1.1 + torch.tanh(Q_feat_adj1))
            # Q_feat2 = Q_feat * (1.1 + torch.tanh(Q_feat_adj2))
            Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))

            scaling = _scaling[N_start:N_end][indices].view(-1)  # [N_num*6]
            scaling = STE_multistep.apply(scaling, Q_scaling, self.get_scaling.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_scaling, min_scaling, max_scaling = encoder_gaussian(scaling, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_scaling_list.append(bit_scaling)
            min_scaling_list.append(min_scaling)
            max_scaling_list.append(max_scaling)
            scaling_list.append(scaling)

            mask = _mask[N_start:N_end][indices]  # {0, 1}  # [N_num, K, 1]
            mask = mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]
            offsets = _grid_offsets[N_start:N_end][indices].view(-1, 3*self.n_offsets).view(-1)  # [N_num*K*3]
            offsets = STE_multistep.apply(offsets, Q_offsets, self._offset.mean())
            offsets[~mask] = 0.0
            torch.cuda.synchronize(); t0 = time.time()
            bit_offsets, min_offsets, max_offsets = encoder_gaussian(offsets[mask], mean_offsets[mask], scale_offsets[mask], Q_offsets[mask], file_name=offsets_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_offsets_list.append(bit_offsets)
            min_offsets_list.append(min_offsets)
            max_offsets_list.append(max_offsets)
            offsets_list.append(offsets)

            torch.cuda.empty_cache()

        # bit_anchor = N * 3 * anchor_round_digits
        bit_anchor = bits_xyz
        # bit_feat1 = sum(bit_feat1_list)
        # bit_feat2 = sum(bit_feat2_list)
        bit_feat_list = [sum(bit_feat) for bit_feat in bit_feat_lists]
        total_bit_feat = sum(bit_feat_list)
        bit2MB_feat_list = [round(bit_feat/bit2MB_scale, 4) for bit_feat in bit_feat_list]
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)


        indices = torch.cat(indices_list, dim=0)
        assert indices.shape[0] == _mask.shape[0]
        mask = _mask[indices]  # {0, 1}
        p = torch.zeros_like(mask).to(torch.float32)
        prob_masks = (mask.sum() / mask.numel()).item()
        p[...] = prob_masks
        bit_masks = encoder((mask * 2 - 1).view(-1), p.view(-1), file_name=masks_b_name)

        show_feat_bits = ' '
        for i in range(self.num_feat_slices):
            show_feat_bits += f'feat{i} {round(bit_feat_list[i]/bit2MB_scale, 4)}, '

        triplane_coder_list, triplane_byte = self.encode_triplane(pre_path_name)

        torch.cuda.synchronize(); t2 = time.time()
        print('encoding time:', t2 - t1)
        print('codec time:', t_codec)
        ftrirate=feat_rate / bit2MB_scale
        log_info = f"\nEncoded sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"total_feat {round(total_bit_feat/bit2MB_scale, 4)}, " \
                   f"{show_feat_bits}" \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Triplane_f {round(triplane_byte/(1024*1024), 4)}," \
                   f"Total {round((bit_anchor + total_bit_feat + bit_scaling + bit_offsets + bit_masks + self.get_mlp_size()[0])/bit2MB_scale+ftrirate.item(), 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"
        rate_set=[round(bit_anchor/bit2MB_scale, 4), round(total_bit_feat/bit2MB_scale, 4)] + bit2MB_feat_list + [round(bit_scaling/bit2MB_scale, 4),round(bit_offsets/bit2MB_scale, 4),round(bit_masks/bit2MB_scale, 4),round(self.get_mlp_size()[0]/bit2MB_scale, 4),ftrirate.item()]
        return [self._anchor.shape[0], N, MAX_batch_size, anchor_infos_list, min_feat_lists, max_feat_lists, min_scaling_list, max_scaling_list, min_offsets_list, max_offsets_list, prob_masks, triplane_coder_list], log_info , rate_set

    @torch.no_grad()
    def conduct_decoding(self, pre_path_name, patched_infos, weighted_mask):
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ...')
        [N_full, N, MAX_batch_size, anchor_infos_list, min_feat_lists, max_feat_lists, min_scaling_list, max_scaling_list, min_offsets_list, max_offsets_list, prob_masks, triplane_coder_list] = patched_infos
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []
        self.decode_triplane(pre_path_name, triplane_coder_list)

        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        p = torch.zeros(size=[N, self.n_offsets, 1], device='cuda').to(torch.float32)
        p[...] = prob_masks
        masks_decoded = decoder(p.view(-1), masks_b_name)  # {-1, 1}
        masks_decoded = (masks_decoded + 1) / 2  # {0, 1}
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)

        mask_anchor = self.get_mask_anchor(weighted_mask)

        # _anchor = self.get_anchor[mask_anchor]
        # _feat = self._anchor_feat[mask_anchor]
        # _grid_offsets = self._offset[mask_anchor]
        # _scaling = self.get_scaling[mask_anchor]
        # _mask = self.get_mask(weighted_mask)[mask_anchor]



        Q_feat_list = []
        Q_scaling_list = []
        Q_offsets_list = []
        
        # anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.pkl')).cuda()
        npz_path = os.path.join(pre_path_name, 'xyz_pcc.bin')
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(parent_dir, 'GausPcgc/best_model_ue_4stage_conv.pt') #   gauspcgc model path
        anchor_decoded = decompress_point_cloud(npz_path,model_path)

        _anchor_int_dec = anchor_decoded['point_cloud'].to('cuda')
        sorted_indices = calculate_morton_order(_anchor_int_dec)
        _anchor_int_dec = _anchor_int_dec[sorted_indices]
        anchor_decoded = _anchor_int_dec * self.voxel_size
        N = anchor_decoded.shape[0]


        for s in range(steps):
            min_feat_per_slice = [min_feat_lists[i][s] for i in range(self.num_feat_slices)]
            max_feat_per_slice = [max_feat_lists[i][s] for i in range(self.num_feat_slices)]
            
            min_scaling = min_scaling_list[s]
            max_scaling = max_scaling_list[s]
            min_offsets = min_offsets_list[s]
            max_offsets = max_offsets_list[s]

            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name_list = []
            for i in range(self.num_feat_slices):
                feat_b_name_list.append(os.path.join(pre_path_name, f'feat{i}.b').replace('.b', f'_{s}.b'))
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2
            indices = torch.tensor(data=range(N_num), device='cuda', dtype=torch.long)  # [N_num]
            anchor_infos = None
            anchor_infos_list.append(anchor_infos)
            # indices_list.append(indices+N_start)

            # anchor_sort = _anchor[N_start:N_end][indices]

            # encode feat
            feat_context, feat_rate = self.feature_net(anchor_decoded[N_start:N_end], itr=-1)
            scales, means, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(feat_context, split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)
            means = list(torch.split(means, split_size_or_sections=self.chcm_slices_list, dim=-1))
            scales = list(torch.split(scales, split_size_or_sections=self.chcm_slices_list, dim=-1))
            Q_feat_list = [Q_feat * (1.1 + torch.tanh(Q_feat_adj.contiguous().repeat(1, ch).view(-1))) for ch in self.chcm_slices_list]
            # feat = _feat[N_start:N_end][indices]  # [N_num*32]

            # _feat_list = list(torch.split(feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
            
            # Q_feat_list.append(Q_feat * (1 + torch.tanh(Q_feat_adj.contiguous())))
            # Q_scaling_list.append(Q_scaling * (1 + torch.tanh(Q_scaling_adj.contiguous())))
            # Q_offsets_list.append(Q_offsets * (1 + torch.tanh(Q_offsets_adj.contiguous())))

            for i in range(self.num_feat_slices):
                if i == 0:
                    mean = means[i].contiguous().view(-1)
                    scale = torch.clamp(scales[i].contiguous().view(-1), min=1e-9)
                    feat = decoder_gaussian(mean, scale, Q_feat_list[i], file_name=feat_b_name_list[i], min_value=min_feat_per_slice[i], max_value=max_feat_per_slice[i])
                    feat = feat.view(N_num, self.chcm_slices_list[i])
                    # temp_feat = _feat_list[i].reshape(-1)
                    # temp_feat = STE_multistep.apply(temp_feat, Q_feat_list[i], _feat.mean())
                    # temp_feat = temp_feat.reshape(N_num, -1)
                    decoded_feat = feat
                    
                    
                else:
                    dmean, dscale = torch.split(self.mlp_chcm_list[i-1](decoded_feat), split_size_or_sections=[self.chcm_slices_list[i], self.chcm_slices_list[i]], dim=-1)
                    mean = means[i] + dmean
                    scale = scales[i] + dscale
                    # if s == 5:
                    #     print(mean[0], scale[0])
                    mean = mean.contiguous().view(-1)
                    scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
                    feat = decoder_gaussian(mean, scale, Q_feat_list[i], file_name=feat_b_name_list[i], min_value=min_feat_per_slice[i], max_value=max_feat_per_slice[i])
                    feat = feat.view(N_num, self.chcm_slices_list[i])
                    # temp_feat = _feat_list[i].reshape(-1)
                    # temp_feat = STE_multistep.apply(temp_feat, Q_feat_list[i], _feat.mean())
                    # temp_feat = temp_feat.reshape(N_num, -1)
                    decoded_feat = torch.cat([decoded_feat, feat], dim=-1)
                    # print(feat_b_name_list[i])
                    # print(torch.mean(feat-temp_feat))
            feat_decoded_list.append(decoded_feat)
            

            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
            if self.chcm_for_scaling:
                dmean_scaling, dscale_scaling = torch.split(self.mlp_chcm_scaling(decoded_feat), split_size_or_sections=[6, 6], dim=-1)
                mean_scaling = mean_scaling + dmean_scaling
                scale_scaling = scale_scaling + dscale_scaling
            mean_scaling = mean_scaling.contiguous().view(-1)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scaling_decoded = decoder_gaussian(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, min_value=min_scaling, max_value=max_scaling)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)

            
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))
            if self.chcm_for_offsets:
                dmean_offsets, dscale_offsets = torch.split(self.mlp_chcm_offsets(decoded_feat), split_size_or_sections=[3*self.n_offsets, 3*self.n_offsets], dim=-1)
                mean_offsets = mean_offsets + dmean_offsets
                scale_offsets = scale_offsets + dscale_offsets
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            masks_tmp = masks_decoded[N_start:N_end].repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
            offsets_decoded_tmp = decoder_gaussian(mean_offsets[masks_tmp], scale_offsets[masks_tmp], Q_offsets[masks_tmp], file_name=offsets_b_name, min_value=min_offsets, max_value=max_offsets)
            offsets_decoded = torch.zeros_like(mean_offsets)
            offsets_decoded[masks_tmp] = offsets_decoded_tmp
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)
            
            
            # mean1 = mean1.contiguous().view(-1)
            # scale1 = torch.clamp(scale1.contiguous().view(-1), min=1e-9)
            # Q_feat1 = Q_feat * (1.1 + torch.tanh(Q_feat_adj1))
            # Q_feat2 = Q_feat * (1.1 + torch.tanh(Q_feat_adj2))
            # feat_decoded1 = decoder_gaussian(mean1, scale1, Q_feat1, file_name=feat1_b_name, min_value=min_feat1, max_value=max_feat1)
            # feat_decoded1 = feat_decoded1.view(N_num, self.feat_dim//2)  # [N_num, 32]
            # dmean2, dscale2 = torch.split(self.mlp_context_from_f1(feat_decoded1), split_size_or_sections=[self.feat_dim//2, self.feat_dim//2], dim=-1)
            # scale2 = scale2 + dscale2
            # mean2 = mean2 + dmean2
            # mean2 = mean2.contiguous().view(-1)
            # scale2 = torch.clamp(scale2.contiguous().view(-1), min=1e-9)
            # # print(mean2.shape, scale2.shape, Q_feat2.shape)
            # feat_decoded2 = decoder_gaussian(mean2, scale2, Q_feat2, file_name=feat2_b_name, min_value=min_feat2, max_value=max_feat2)
            # feat_decoded2 = feat_decoded2.view(N_num, self.feat_dim//2)  # [N_num, 32]
            # feat_decoded = torch.cat([feat_decoded1, feat_decoded2], dim=-1)  # [N_num, 64]
            # feat_decoded_list.append(feat_decoded)
            
            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        torch.cuda.synchronize(); t2 = time.time()
        print('decoding time:', t2 - t1)

        # fill back N_full
        _anchor = torch.zeros(size=[N, 3], device='cuda')
        _anchor_feat = torch.zeros(size=[N, self.feat_dim], device='cuda')
        _offset = torch.zeros(size=[N, self.n_offsets, 3], device='cuda')
        _scaling = torch.zeros(size=[N, 6], device='cuda')
        _mask = torch.zeros(size=[N, self.n_offsets, 1], device='cuda')

        _anchor[:N] = anchor_decoded
        _anchor_feat[:N] = feat_decoded
        _offset[:N] = offsets_decoded
        _scaling[:N] = scaling_decoded
        _mask[:N] = masks_decoded

        print('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        # assert self._anchor_feat.shape == _anchor_feat.shape
        
        self._anchor_feat = nn.Parameter(_anchor_feat)
        # assert self._offset.shape == _offset.shape
        self._offset = nn.Parameter(_offset)
        # If change the following attributes, decoded_version must be set True
        self.decoded_version = True
        # assert self.get_anchor.shape == _anchor.shape
        self._anchor = nn.Parameter(_anchor)
        # assert self._scaling.shape == _scaling.shape
        self._scaling = nn.Parameter(_scaling)
        # assert self._mask.shape == _mask.shape
        self._mask = nn.Parameter(_mask)



        print('Parameters are successfully replaced by decoded ones!')

        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        return log_info
    
    def encode_triplane(self, pre_path_name):
        encode_start_time = time.time()
        self.feature_net.attribute_net.grid.non_zero_pixel_ctx_index = self.feature_net.attribute_net.grid.non_zero_pixel_ctx_index.to('cpu')
        self.feature_net.attribute_net.grid = self.feature_net.attribute_net.grid.to('cpu')
        temp_output = self.feature_net.attribute_net.grid.grid_encode_forward(get_proba_param = True)


        range_coder_latent_list = []
        ac_max_val_list = []
        for latent in temp_output.get('latent'):
            ac_max_val_latent = get_ac_max_val_latent(latent)
            range_coder_latent = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent, Q_EXP_SCALE)
            range_coder_latent_list.append(range_coder_latent)
            ac_max_val_list.append(range_coder_latent.AC_MAX_VAL)
            
        range_coder_latent_list2 = []
        ac_max_val_list2 = []
        for latent in temp_output.get('latent2'):
            ac_max_val_latent = get_ac_max_val_latent(latent)
            range_coder_latent = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent, Q_EXP_SCALE)
            range_coder_latent_list2.append(range_coder_latent)
            ac_max_val_list2.append(range_coder_latent.AC_MAX_VAL)
            

        range_coder_latent_list3 = []
        ac_max_val_list3 = []
        for latent in temp_output.get('latent3'):
            ac_max_val_latent = get_ac_max_val_latent(latent)
            range_coder_latent = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent, Q_EXP_SCALE)
            range_coder_latent_list3.append(range_coder_latent)
            ac_max_val_list3.append(range_coder_latent.AC_MAX_VAL)



        encoder_output = self.feature_net.attribute_net.grid.grid_encode_forward(get_proba_param = True, AC_MAX_VAL = ac_max_val_list, AC_MAX_VAL2 = ac_max_val_list2, AC_MAX_VAL3 = ac_max_val_list3)
        xy_bitstream_path = os.path.join(pre_path_name, 'xy_bitstream')
        n_bytes_per_latent = []
        for j in range(len(encoder_output.get('latent'))):
            current_mu = encoder_output.get('mu')[j]
            current_scale = encoder_output.get('scale')[j]
            current_scale = torch.round(current_scale*Q_EXP_SCALE)/Q_EXP_SCALE
            current_scale = torch.clamp(current_scale,  min=1/Q_EXP_SCALE)
            current_y = encoder_output.get('latent')[j]
            cur_latent_bitstream = f'{xy_bitstream_path}_{j}.bin'
            range_coder_latent_list[j].encode(
                cur_latent_bitstream,
                current_y.flatten().cpu(),
                current_mu.flatten().cpu(),
                current_scale.flatten().cpu(),
                (self.feature_net.attribute_net.grid.output_dim, current_y.size()[-2], current_y.size()[-1]),
            )
            n_bytes_per_latent.append(os.path.getsize(cur_latent_bitstream))

        yz_bitstream_path = os.path.join(pre_path_name, 'yz_bitstream')
        n_bytes_per_latent = []
        for j in range(len(encoder_output.get('latent2'))):
            current_mu = encoder_output.get('mu2')[j]
            current_scale = encoder_output.get('scale2')[j]
            current_scale = torch.round(current_scale*Q_EXP_SCALE)/Q_EXP_SCALE
            current_scale = torch.clamp(current_scale,  min=1/Q_EXP_SCALE)
            current_y = encoder_output.get('latent2')[j]
            cur_latent_bitstream = f'{yz_bitstream_path}_{j}.bin'
            range_coder_latent_list2[j].encode(
                cur_latent_bitstream,
                current_y.flatten().cpu(),
                current_mu.flatten().cpu(),
                current_scale.flatten().cpu(),
                (self.feature_net.attribute_net.grid.output_dim, current_y.size()[-2], current_y.size()[-1]),
            )
            n_bytes_per_latent.append(os.path.getsize(cur_latent_bitstream))

        zx_bitstream_path = os.path.join(pre_path_name, 'zx_bitstream')
        n_bytes_per_latent = []
        for j in range(len(encoder_output.get('latent3'))):
            current_mu = encoder_output.get('mu3')[j]
            current_scale = encoder_output.get('scale3')[j]
            current_scale = torch.round(current_scale*Q_EXP_SCALE)/Q_EXP_SCALE
            current_scale = torch.clamp(current_scale,  min=1/Q_EXP_SCALE)
            current_y = encoder_output.get('latent3')[j]
            cur_latent_bitstream = f'{zx_bitstream_path}_{j}.bin'
            range_coder_latent_list3[j].encode(
                cur_latent_bitstream,
                current_y.flatten().cpu(),
                current_mu.flatten().cpu(),
                current_scale.flatten().cpu(),
                (self.feature_net.attribute_net.grid.output_dim, current_y.size()[-2], current_y.size()[-1]),
            )
            n_bytes_per_latent.append(os.path.getsize(cur_latent_bitstream))

        
        # Write the header
        bitstream_path = 'bitstream.bin'
        subprocess.call(f'rm -f {bitstream_path}', shell=True)
        
        real_rate_byte = 0
        for j in range(len(encoder_output.get('latent'))):
            cur_latent_bitstream = f'{xy_bitstream_path}_{j}.bin'
            real_rate_byte += os.path.getsize(cur_latent_bitstream)
  
        for j in range(len(encoder_output.get('latent2'))):
            cur_latent_bitstream = f'{yz_bitstream_path}_{j}.bin'
            real_rate_byte += os.path.getsize(cur_latent_bitstream)

        for j in range(len(encoder_output.get('latent3'))):
            cur_latent_bitstream = f'{zx_bitstream_path}_{j}.bin'
            real_rate_byte += os.path.getsize(cur_latent_bitstream)
        
    
        
        encode_end_time = time.time()
        print('Encoding time:', encode_end_time - encode_start_time)

        return [ac_max_val_list, ac_max_val_list2, ac_max_val_list3, range_coder_latent_list, range_coder_latent_list2, range_coder_latent_list3], real_rate_byte

    def decode_triplane(self, pre_path_name, ax_max_val_lists):
        decode_start_time = time.time()

        ac_max_val_list, ac_max_val_list2, ac_max_val_list3, range_coder_latent_list, range_coder_latent_list2, range_coder_latent_list3 = ax_max_val_lists[0], ax_max_val_lists[1], ax_max_val_lists[2], ax_max_val_lists[3], ax_max_val_lists[4], ax_max_val_lists[5]
        decoded_triplane = nn.ModuleList()

        decoded_latents = [nn.ParameterList() for _ in range(len(self.feature_net.attribute_net.grid.multiscale_res_multipliers))]
        self.feature_net.attribute_net.grid.arm = self.feature_net.attribute_net.grid.arm.to('cuda')
        self.feature_net.attribute_net.grid.arm2 = self.feature_net.attribute_net.grid.arm2.to('cuda')
        self.feature_net.attribute_net.grid.arm3 = self.feature_net.attribute_net.grid.arm3.to('cuda')

        for j in range(2):
            decoded_y = []
            coo_combs = list(itertools.combinations(range(3), 2))
                # print(coo_combs)
            hw = [self.feature_net.attribute_net.grid.resolutions[cc] for cc in coo_combs[1][::-1]]

            h_grid, w_grid = hw[0] * self.feature_net.attribute_net.grid.multiscale_res_multipliers[j], hw[1] * self.feature_net.attribute_net.grid.multiscale_res_multipliers[j]
            
            n_ctx_rowcol = 2
            mask_size = 2 * n_ctx_rowcol + 1
            pad = n_ctx_rowcol

            range_coder = range_coder_latent_list[j]
            xy_bitstream_path = os.path.join(pre_path_name, 'xy_bitstream')
        
            range_coder.load_bitstream(f'{xy_bitstream_path}_{j}.bin')
            coding_order = range_coder.generate_coding_order(
                (self.feature_net.attribute_net.grid.output_dim, h_grid, w_grid), n_ctx_rowcol
            )
            flat_coding_order = coding_order.flatten().contiguous()
            flat_coding_order_np = flat_coding_order.detach().cpu().numpy()
            flat_index_coding_order_np = np.argsort(flat_coding_order_np, kind='stable')
            flat_index_coding_order = torch.from_numpy(flat_index_coding_order_np).long().to(flat_coding_order.device)
            _, occurrence_coding_order = torch.unique(flat_coding_order, return_counts=True)
            current_y = torch.zeros((1, self.feature_net.attribute_net.grid.output_dim, h_grid, w_grid), device='cpu')

            # Compute the 1d offset of indices to form the context
            offset_index_arm = compute_offset(current_y, mask_size)

            # pad and flatten and current y to perform everything in 1d
            current_y = F.pad(current_y, (pad, pad, pad, pad), mode='constant', value=0.)
            current_y = current_y.flatten().contiguous()

            # Pad the coding order with unreachable values (< 0) to mimic the current_y padding
            # then flatten it as we want to index current_y which is a 1d tensor
            coding_order = F.pad(coding_order, (pad, pad, pad, pad), mode='constant', value=-1)
            coding_order = coding_order.flatten().contiguous()

            # Count the number of decoded values to iterate within the
            # flat_index_coding_order array.
            cnt = 0
            n_ctx_row_col = n_ctx_rowcol
            for index_coding in range(flat_coding_order.max() + 1):
                cur_context = fast_get_neighbor(
                    current_y, mask_size, offset_index_arm,
                    flat_index_coding_order[cnt: cnt + occurrence_coding_order[index_coding]],
                    w_grid, self.feature_net.attribute_net.grid.output_dim
                )
                
                # ----- From now: run on CPU
                # Compute proba param from context
                
                cur_raw_proba_param = self.feature_net.attribute_net.grid.arm(cur_context.cuda())
                cur_raw_proba_param = cur_raw_proba_param.cpu()
                cur_mu, cur_scale = get_mu_scale(cur_raw_proba_param)
                cur_scale = torch.round(cur_scale*Q_EXP_SCALE)/Q_EXP_SCALE
                cur_scale = torch.clamp(cur_scale, min=1/Q_EXP_SCALE)
                # Decode and store the value at the proper location within current_y
                x_delta = n_ctx_row_col+1
                if index_coding < w_grid:
                    start_y = 0
                    start_x = index_coding
                else:
                    start_y = (index_coding - w_grid) // x_delta + 1
                    start_x = w_grid - x_delta + (index_coding - w_grid) % x_delta

                x = range_coder.decode(cur_mu, cur_scale)
                current_y[
                    [
                        coding_order == index_coding
                    ]
                ] = x
                # Increment the counter of loaded value
                cnt += occurrence_coding_order[index_coding]

            # Reshape y as a 4D grid, and remove padding
            current_y = current_y.reshape(1, self.feature_net.attribute_net.grid.output_dim, h_grid + 2 * pad, w_grid + 2 * pad)
            current_y = current_y[:, :, pad:-pad, pad:-pad]
            current_y = current_y / 2**4
            
            decoded_latents[j].append(nn.Parameter(current_y))


        for j in range(2):
            decoded_y = []
            coo_combs = list(itertools.combinations(range(3), 2))
            hw = [self.feature_net.attribute_net.grid.resolutions[cc] for cc in coo_combs[1][::-1]]

            h_grid, w_grid = hw[0] * self.feature_net.attribute_net.grid.multiscale_res_multipliers[j], hw[1] * self.feature_net.attribute_net.grid.multiscale_res_multipliers[j]
            
            n_ctx_rowcol = 2
            mask_size = 2 * n_ctx_rowcol + 1
            # How many padding pixels we need
            pad = n_ctx_rowcol

            range_coder = range_coder_latent_list2[j]
            bitstream_path = os.path.join(pre_path_name, 'yz_bitstream')
            range_coder.load_bitstream(f'{bitstream_path}_{j}.bin')
            
            coding_order = range_coder.generate_coding_order(
                (self.feature_net.attribute_net.grid.output_dim, h_grid, w_grid), n_ctx_rowcol
            )

            # --------- Wave front coding order
            # With n_ctx_rowcol = 1, coding order is something like
            # 0 1 2 3 4 5 6
            # 1 2 3 4 5 6 7
            # 2 3 4 5 6 7 8
            # which indicates the order of the wavefront coding process.
            # print(coding_order)
            # exit()
            flat_coding_order = coding_order.flatten().contiguous()

            # flat_index_coding_is an array with [H_grid * W_grid] elements.
            # It indicates the index of the different coding cycle i.e.
            # the concatenation of the following list:
            #   [indices for which flat_coding_order == 0]
            #   [indices for which flat_coding_order == 1]
            #   [indices for which flat_coding_order == 2]
            # flat_coding_order = flat_coding_order.to('cpu')
            # print(flat_coding_order)
            flat_coding_order_np = flat_coding_order.detach().cpu().numpy()
            flat_index_coding_order_np = np.argsort(flat_coding_order_np, kind='stable')
            flat_index_coding_order = torch.from_numpy(flat_index_coding_order_np).long().to(flat_coding_order.device)
            # flat_index_coding_order = flat_coding_order.argsort(kind='stable')

            # occurrence_coding_order gives us the number of values to be decoded
            # at each wave-front cycle i.e. the number of values whose coding
            # order is i.
            # occurrence_coding_order[i] = number of values to decode at step i
            _, occurrence_coding_order = torch.unique(flat_coding_order, return_counts=True)
            # print(occurrence_coding_order)

            # Current latent grid without padding
            current_y = torch.zeros((1, self.feature_net.attribute_net.grid.output_dim, h_grid, w_grid), device='cpu')

            # Compute the 1d offset of indices to form the context
            offset_index_arm = compute_offset(current_y, mask_size)

            # pad and flatten and current y to perform everything in 1d
            current_y = F.pad(current_y, (pad, pad, pad, pad), mode='constant', value=0.)
            current_y = current_y.flatten().contiguous()

            # Pad the coding order with unreachable values (< 0) to mimic the current_y padding
            # then flatten it as we want to index current_y which is a 1d tensor
            coding_order = F.pad(coding_order, (pad, pad, pad, pad), mode='constant', value=-1)
            coding_order = coding_order.flatten().contiguous()

            # Count the number of decoded values to iterate within the
            # flat_index_coding_order array.
            cnt = 0
            n_ctx_row_col = n_ctx_rowcol
            for index_coding in range(flat_coding_order.max() + 1):
                cur_context = fast_get_neighbor(
                    current_y, mask_size, offset_index_arm,
                    flat_index_coding_order[cnt: cnt + occurrence_coding_order[index_coding]],
                    w_grid, self.feature_net.attribute_net.grid.output_dim
                )
                # print('b', cur_context[:, -1].flatten(), cur_context[:, -1].flatten().shape)

                # ----- From now: run on CPU
                # Compute proba param from context
                # print('cxt', torch.min(cur_context[-1]), torch.max(cur_context[-1]))
                cur_raw_proba_param = self.feature_net.attribute_net.grid.arm2(cur_context.cuda())
                cur_raw_proba_param = cur_raw_proba_param.cpu()
                # cur_raw_proba_param = self.feature_net.attribute_net.grid.arm2(cur_context.cpu())
                cur_mu, cur_scale = get_mu_scale(cur_raw_proba_param)
                cur_scale = torch.round(cur_scale*Q_EXP_SCALE)/Q_EXP_SCALE
                cur_scale = torch.clamp(cur_scale, min=1/Q_EXP_SCALE)
                # cur_scale = torch.round(cur_scale)
                # Decode and store the value at the proper location within current_y
                x_delta = n_ctx_row_col+1
                if index_coding < w_grid:
                    start_y = 0
                    start_x = index_coding
                else:
                    start_y = (index_coding - w_grid) // x_delta + 1
                    start_x = w_grid - x_delta + (index_coding - w_grid) % x_delta

                x = range_coder.decode(cur_mu, cur_scale)
                current_y[
                    [
                        coding_order == index_coding
                    ]
                ] = x
                # Increment the counter of loaded value
                cnt += occurrence_coding_order[index_coding]

            # Reshape y as a 4D grid, and remove padding
            current_y = current_y.reshape(1, self.feature_net.attribute_net.grid.output_dim, h_grid + 2 * pad, w_grid + 2 * pad)
            current_y = current_y[:, :, pad:-pad, pad:-pad]
            current_y = current_y / 2**4
            
            decoded_latents[j].append(nn.Parameter(current_y))

        
        for j in range(2):
            decoded_y = []
            coo_combs = list(itertools.combinations(range(3), 2))
            hw = [self.feature_net.attribute_net.grid.resolutions[cc] for cc in coo_combs[1][::-1]]

            h_grid, w_grid = hw[0] * self.feature_net.attribute_net.grid.multiscale_res_multipliers[j], hw[1] * self.feature_net.attribute_net.grid.multiscale_res_multipliers[j]
            
            n_ctx_rowcol = 2
            mask_size = 2 * n_ctx_rowcol + 1
            # How many padding pixels we need
            pad = n_ctx_rowcol

            range_coder = range_coder_latent_list3[j]
            bitstream_path = os.path.join(pre_path_name, 'zx_bitstream')
            range_coder.load_bitstream(f'{bitstream_path}_{j}.bin')
            
            coding_order = range_coder.generate_coding_order(
                (self.feature_net.attribute_net.grid.output_dim, h_grid, w_grid), n_ctx_rowcol
            )

            # --------- Wave front coding order
            # With n_ctx_rowcol = 1, coding order is something like
            # 0 1 2 3 4 5 6
            # 1 2 3 4 5 6 7
            # 2 3 4 5 6 7 8
            # which indicates the order of the wavefront coding process.
            flat_coding_order = coding_order.flatten().contiguous()

            # flat_index_coding_is an array with [H_grid * W_grid] elements.
            # It indicates the index of the different coding cycle i.e.
            # the concatenation of the following list:
            #   [indices for which flat_coding_order == 0]
            #   [indices for which flat_coding_order == 1]
            #   [indices for which flat_coding_order == 2]
            # flat_coding_order = flat_coding_order.to('cpu')
            flat_coding_order_np = flat_coding_order.detach().cpu().numpy()
            flat_index_coding_order_np = np.argsort(flat_coding_order_np, kind='stable')
            flat_index_coding_order = torch.from_numpy(flat_index_coding_order_np).long().to(flat_coding_order.device)
            
            # occurrence_coding_order gives us the number of values to be decoded
            # at each wave-front cycle i.e. the number of values whose coding
            # order is i.
            # occurrence_coding_order[i] = number of values to decode at step i
            _, occurrence_coding_order = torch.unique(flat_coding_order, return_counts=True)
            
            # Current latent grid without padding
            current_y = torch.zeros((1, self.feature_net.attribute_net.grid.output_dim, h_grid, w_grid), device='cpu')

            # Compute the 1d offset of indices to form the context
            offset_index_arm = compute_offset(current_y, mask_size)

            # pad and flatten and current y to perform everything in 1d
            current_y = F.pad(current_y, (pad, pad, pad, pad), mode='constant', value=0.)
            current_y = current_y.flatten().contiguous()

            # Pad the coding order with unreachable values (< 0) to mimic the current_y padding
            # then flatten it as we want to index current_y which is a 1d tensor
            coding_order = F.pad(coding_order, (pad, pad, pad, pad), mode='constant', value=-1)
            coding_order = coding_order.flatten().contiguous()

            # Count the number of decoded values to iterate within the
            # flat_index_coding_order array.
            cnt = 0
            n_ctx_row_col = n_ctx_rowcol
            for index_coding in range(flat_coding_order.max() + 1):
                cur_context = fast_get_neighbor(
                    current_y, mask_size, offset_index_arm,
                    flat_index_coding_order[cnt: cnt + occurrence_coding_order[index_coding]],
                    w_grid, self.feature_net.attribute_net.grid.output_dim
                )
                cur_raw_proba_param = self.feature_net.attribute_net.grid.arm3(cur_context.cuda())
                cur_raw_proba_param = cur_raw_proba_param.cpu()
                cur_mu, cur_scale = get_mu_scale(cur_raw_proba_param)
                cur_scale = torch.round(cur_scale*Q_EXP_SCALE)/Q_EXP_SCALE
                cur_scale = torch.clamp(cur_scale, min=1/Q_EXP_SCALE)
                x_delta = n_ctx_row_col+1
                if index_coding < w_grid:
                    start_y = 0
                    start_x = index_coding
                else:
                    start_y = (index_coding - w_grid) // x_delta + 1
                    start_x = w_grid - x_delta + (index_coding - w_grid) % x_delta

                x = range_coder.decode(cur_mu, cur_scale)
                current_y[
                    [
                        coding_order == index_coding
                    ]
                ] = x
                cnt += occurrence_coding_order[index_coding]

            current_y = current_y.reshape(1, self.feature_net.attribute_net.grid.output_dim, h_grid + 2 * pad, w_grid + 2 * pad)
            current_y = current_y[:, :, pad:-pad, pad:-pad]
            current_y = current_y / 2**4

            decoded_latents[j].append(nn.Parameter(current_y))
        
        
        for decoded_latent in decoded_latents:
            decoded_triplane.append(decoded_latent)

        decode_end_time = time.time()

        print('Triplane decoding time:', decode_end_time - decode_start_time)
        

        

        self.feature_net.attribute_net.grid.grids = decoded_triplane
        self.feature_net.attribute_net.grid = self.feature_net.attribute_net.grid.cuda()
        self.feature_net.attribute_net.grid.grids = self.feature_net.attribute_net.grid.grids.cuda()
        self.feature_net.attribute_net.grid.non_zero_pixel_ctx_index = self.feature_net.attribute_net.grid.non_zero_pixel_ctx_index.cuda()





