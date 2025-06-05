import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.synthesis import UniformNoiseQuantizer,STEQuantizer
from scene.arm import ArmMLP, get_flat_latent_and_context, compute_rate
import os
from torch import Tensor, nn
from typing_extensions import TypedDict

class EncoderOutput(TypedDict):
    """Define the dictionary containing COOL-CHIC encoder output as a type."""
    rate_y: Tensor                     # Rate [1]
    mu: Optional[List[Tensor]]         # List of N [H_i, W_i] tensors, mu for each latent grid (can be None)
    scale: Optional[List[Tensor]]      # List of N [H_i, W_i] tensors, scale for each latent grid (can be None)
    latent: Optional[List[Tensor]]     # List of N [H_i, W_i] tensors, each latent grid (can be None)




def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0

def con_normalize_aabb(pts, aabb, con_aabb):
    scale = (aabb[1] - aabb[0]) / (con_aabb[1] - con_aabb[0])
    normalized_pts = (pts - con_aabb[0]) * scale + aabb[0]
    return normalized_pts

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = -0.01,
        b: float = 0.01):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            level = 1
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        # print(scale_id)
        interp_space = 1.
        coo_dict = []
        for ci, coo_comb in enumerate(coo_combs):
            
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            
            coo_dict.append(interp_out_plane)
            interp_space = interp_space * interp_out_plane

        triplane_result = torch.cat(coo_dict, dim=1) 
        concat_tri=True
        
        if concat_features:
            if concat_tri:
                if scale_id != 0:
                    if level > scale_id:
                        multi_scale_interp.append(triplane_result + multi_scale_interp[-1])
                    else:
                        multi_scale_interp.append(multi_scale_interp[-1])
                else:
                    multi_scale_interp.append(triplane_result)
            else:
                if scale_id != 0:
                    multi_scale_interp.append(interp_space + multi_scale_interp[-1])
                else:
                    multi_scale_interp.append(interp_space)
        else:
            if concat_tri:
                multi_scale_interp = multi_scale_interp + triplane_result
            else:
                multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    
    return multi_scale_interp


class TriPlaneField(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires,
        contract,
        comp_iter,
        layers_arm: List = [16, 16, 16, 16],
        n_ctx_rowcol = 2,
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        self.total_bits = 0
        config = self.grid_config[0].copy()
        self.resolutions = config["resolution"]

        self.init_itr=comp_iter
        self.out_channels = config["output_coordinate_dim"]

        self.rotation_matrix = nn.Parameter(torch.eye(3), requires_grad=True)
        self.pca_mean = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.variance = nn.Parameter(torch.ones(3), requires_grad=True)
        self.con_aabb = nn.Parameter(torch.ones([2,3]),requires_grad=True)
        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:]
            ]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1] * 3
            else:
                self.feat_dim = gp[-1].shape[1] * 3
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
        self.quantizer = UniformNoiseQuantizer()
        self.ste_quantizer = STEQuantizer()
        self.log_2_encoder_gains = nn.Parameter(
            torch.arange(0., 5), requires_grad=True
        )
        non_zero_pixel_ctx = int((5 ** 2 - 1) / 2)
        self.non_zero_pixel_ctx_index = torch.arange(0, non_zero_pixel_ctx).cuda()
        self.arm = torch.jit.script(ArmMLP(non_zero_pixel_ctx, layers_arm))
        self.arm2 = torch.jit.script(ArmMLP(non_zero_pixel_ctx, layers_arm))
        self.arm3 = torch.jit.script(ArmMLP(non_zero_pixel_ctx, layers_arm))
        
        self.contract = contract
    
        self.num_param = (config["resolution"][-1] ** 2) * 1.5 * config["output_coordinate_dim"]
        self.n_ctx_rowcol = n_ctx_rowcol
        self.output_dim = config["output_coordinate_dim"]
        
        self.mask_size = 2 * n_ctx_rowcol + 1

    def set_aabb(self,rotated_points,xyz_max, xyz_min):
        xyz_max = xyz_max.flatten().cpu().detach().numpy()
        xyz_min = xyz_min.flatten().cpu().detach().numpy()
        if self.contract == True:
            pts = self.contract_to_unisphere(rotated_points.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
            xyz_min = torch.min(pts, dim=0, keepdim=True)[0].detach()
            xyz_max = torch.max(pts, dim=0, keepdim=True)[0].detach()
            self.con_aabb = nn.Parameter(torch.cat((xyz_min, xyz_max), dim=0),requires_grad=True)
            print("after contract:",xyz_min,xyz_max)
            xyz_max = [1.0, 1.0, 1.0]
            xyz_min = [-1.0, -1.0, -1.0]
        aabb = torch.tensor([
            xyz_min,
            xyz_max
        ], device='cuda')
        self.aabb = nn.Parameter(aabb,requires_grad=True)
        print("Voxel Plane: set aabb=",self.aabb)

    def set_rotation_matrix(self, rotation_matrix, mean, variance):
        self.rotation_matrix = nn.Parameter(rotation_matrix, requires_grad=False)
        self.pca_mean = nn.Parameter(mean, requires_grad=False)
        self.variance = nn.Parameter(variance, requires_grad=False)

        
    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None, itr=-1):
        """Computes and returns the densities."""
        # if itr==-1:
        #     print('iter=-1')
        grids_hat = self.grids
        # if itr != 4:
        #     print(itr)
        if itr > self.init_itr or itr == -1:
            grids_hat = []
            sent_latent = []
            sent_latent2 = []
            sent_latent3 = []
            for i, cur_latent in enumerate(self.grids):
                temp = []
                ii=0
                for latent in cur_latent:
                    quant_step = 2 ** 4
                    quant_latent = self.quantizer.apply(
                    latent * quant_step, # Apply Q. step
                    False if itr == -1 else True                                           # Noise if needed
                    )
                    if ii==0:
                        sent_latent.append(quant_latent)
                    elif ii==1:
                        sent_latent2.append(quant_latent)
                    elif ii==2:
                        sent_latent3.append(quant_latent)
                    ii = ii + 1
                    round_latent = self.ste_quantizer.apply(
                    latent * quant_step, # Apply Q. step
                    False if itr == -1 else True                                           # Noise if needed
                    )
                    temp.append(round_latent / quant_step)
                grids_hat.append(temp)
                
            
            # print(sent_latent[0].get_device(), self.non_zero_pixel_ctx_index)
            flat_latent, flat_context = get_flat_latent_and_context(
                sent_latent, 5, self.non_zero_pixel_ctx_index
            )
            
            raw_proba_param = self.arm(flat_context)
            rate_y, _, _ = compute_rate(flat_latent, raw_proba_param)
            flat_latent2, flat_context2 = get_flat_latent_and_context(
                sent_latent2, 5, self.non_zero_pixel_ctx_index
            )
            
            raw_proba_param2 = self.arm2(flat_context2)
            rate_y2, _, _ = compute_rate(flat_latent2, raw_proba_param2)
            
            flat_latent3, flat_context3 = get_flat_latent_and_context(
                sent_latent3, 5, self.non_zero_pixel_ctx_index
            )
            
            raw_proba_param3 = self.arm3(flat_context3)
            rate_y3, _, _ = compute_rate(flat_latent3, raw_proba_param3)
            self.total_bits = rate_y.sum()+rate_y2.sum()+rate_y3.sum()
            

        if self.contract == True:
            pts = con_normalize_aabb(pts, self.aabb,self.con_aabb)
        else: 
            pts = normalize_aabb(pts, self.aabb)
        
        # pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        

        pts = pts.reshape(-1, pts.shape[-1])
        if itr > 13000 or itr == -1:
            current_level = 2
        else:
            current_level = 2
        features = interpolate_ms_features(
            pts, ms_grids=grids_hat,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None, level=current_level)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)

        if itr > self.init_itr or itr == -1:    
            return features, self.total_bits
        else:
            return features, 0

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None, itr=-1):
        if self.contract:
            pts = pts - self.pca_mean
            pts = torch.matmul(pts, self.rotation_matrix.detach())
            pts = pts / self.variance.detach()

            pts = self.contract_to_unisphere(pts.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
    
        features = self.get_density(pts, itr=itr)

        return features
    
    def save_latent(self, save_path, itr):
        """Computes and returns the densities."""
        # if itr==-1:
        #     print('iter=-1')
        grids_hat = self.grids
        # if itr != 4:
        #     print(itr)
        if 1:
            grids_hat = []
            sent_latent = []
            sent_latent2 = []
            sent_latent3 = []
            for i, cur_latent in enumerate(self.grids):
                temp = []
                ii=0
                for latent in cur_latent:
                    quant_step = 2 ** 4
                    # quant_latent = self.quantizer.apply(
                    # latent * quant_step, # Apply Q. step
                    # False                                           # Noise if needed
                    # )
                    # if ii==0:
                    #     sent_latent.append(quant_latent)
                    # elif ii==1:
                    #     sent_latent2.append(quant_latent)
                    # elif ii==2:
                    #     sent_latent3.append(quant_latent)
                    # ii = ii + 1
                    round_latent = self.ste_quantizer.apply(
                    latent * quant_step, # Apply Q. step
                    False                                           # Noise if needed
                    )
                    temp.append(round_latent)
                grids_hat.append(temp)

            save_path = os.path.join(save_path, f'grids_hat_{itr}.pth')
            torch.save(grids_hat, save_path)
    
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
        
    def grid_encode_forward(self,
            get_proba_param: bool = False,
            use_ste_quant: bool = True,
            AC_MAX_VAL: list = None,
            AC_MAX_VAL2: list = None,
            AC_MAX_VAL3: list = None,
        )->EncoderOutput:

        self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index

        # ====================== Get sent latent codes ====================== #
        # Two different types of quantization. quantize() function uses the usual
        # noise addition proxy if self.training is True and the actual round
        # otherwise.
        # if use_ste_quant the straight-through estimator is used i.e. actual
        # quantization in the forward pass and gradient set to one in the backward.
        quantizer = self.ste_quantizer if use_ste_quant else self.uniform_noise_quantizer
        sent_latent = []
        sent_latent2 = []
        sent_latent3 = []
        for i, cur_latent in enumerate(self.grids):
            temp = []
            ii=0
            for latent in cur_latent:
                # print('assd', latent.size())
                quant_step = 2 ** 4
                quant_latent = self.quantizer.apply(
                latent * quant_step, # Apply Q. step
                False                                           # Noise if needed
                )
                if ii==0:
                    # print('assd', quant_latent.size())
                    sent_latent.append(quant_latent)
                elif ii==1:
                    # print('assd2', quant_latent.size())
                    sent_latent2.append(quant_latent)
                elif ii==2:
                    # print('assd3', quant_latent.size())
                    sent_latent3.append(quant_latent)
                ii = ii + 1
                
            
        # Clamp latent if we need to write a bitstream
        if AC_MAX_VAL != None:
            sent_latent = [
                torch.clamp(cur_latent, -AC_MAX_VAL[i], AC_MAX_VAL[i] + 1)
                for i, cur_latent in enumerate(sent_latent)
            ]
        if AC_MAX_VAL2 != None:
            sent_latent2 = [
                torch.clamp(cur_latent, -AC_MAX_VAL2[i], AC_MAX_VAL2[i] + 1)
                for i, cur_latent in enumerate(sent_latent2)
            ]
        if AC_MAX_VAL3 != None:
            sent_latent3 = [
                torch.clamp(cur_latent, -AC_MAX_VAL3[i], AC_MAX_VAL3[i] + 1)
                for i, cur_latent in enumerate(sent_latent3)
            ]
        # ====================== Get sent latent codes ====================== #

        # sent_latent
        flat_latents = []
        flat_contexts = []
        raw_proba_params = []
        rate_ys = []
        # print('asd', sent_latent[0].size())
        for i in range(len(sent_latent)):
            flat_latent, flat_context = get_flat_latent_and_context(
                (sent_latent[i]).unsqueeze(0), 5, self.non_zero_pixel_ctx_index
            )
            raw_proba_param = self.arm(flat_context)
            rate_y, _, _ = compute_rate(flat_latent, raw_proba_param)
            rate_ys.append(rate_y)
            flat_latents.append(flat_latent)
            flat_contexts.append(flat_context)
            raw_proba_params.append(raw_proba_param)
        
        rate_y1 = sum([rate_y.sum() for rate_y in rate_ys])
        # print('sent latent3', rate_y1)

        if get_proba_param:
            

            # Prepare lists to accommodate the visualisations
            mus = []
            scales = []
            latents = []

            # for i in range(len(sent_latent)):
            #     _, flat_mu_y, flat_scale_y = compute_rate(flat_latents[i], raw_proba_params[i])
            #     # print(flat_mu_y.size(), flat_scale_y.size())
            #     mu = []
            #     scale = []
            #     latent = []

            #     # "Pointer" for the reading of the 1D scale, mu and rate
            #     cnt = 0
            #     for j, _ in enumerate(sent_latent[i][0]):
                    
            #         h_i, w_i = sent_latent[i].size()[-2:]
            #         # print(sent_latent[0][0, i].size())
            #         # print(h_i, w_i, i)

            #         # Scale, mu and rate are 1D tensors where the N latent grids
            #         # are flattened together. As such we have to read the appropriate
            #         # number of values in this 1D vector to reconstruct the i-th grid in 2D
            #         mu_i, scale_i = [
            #             # Read h_i * w_i values starting from cnt
            #             tmp[cnt: cnt + (h_i * w_i)].view((h_i, w_i))
            #             for tmp in [flat_mu_y, flat_scale_y]
            #         ]
            #         # print(i, mu_i.size())

            #         cnt += h_i * w_i
            #         mu.append(mu_i)
            #         scale.append(scale_i)
            #         latent.append(sent_latent[i][0, j].view(h_i, w_i))

            #     mus.append(mu)
            #     scales.append(scale)
            #     latents.append(latent)

            for i in range(len(sent_latent)):
                _, flat_mu_y, flat_scale_y = compute_rate(flat_latents[i], raw_proba_params[i])
                # print(flat_mu_y.size(), flat_scale_y.size())
                mu = []
                scale = []
                latent = []

                # "Pointer" for the reading of the 1D scale, mu and rate
                cnt = 0
                h_i, w_i = sent_latent[i].size()[-2:]
                mu = flat_mu_y.view(-1, h_i, w_i)
                scale = flat_scale_y.view(-1, h_i, w_i)
                latent = sent_latent[i][0].view(-1, h_i, w_i)

                print(i, mu.size(), scale.size(), latent.size())

                mus.append(mu)
                scales.append(scale)
                latents.append(latent)
            # print('aa', latents[0][0].size())
            # print(mus[0][0].size())
            # print(mus[1][0].size())


        # sent_latent2
        flat_latents = []
        flat_contexts = []
        raw_proba_params = []
        rate_ys = []
        for i in range(len(sent_latent2)):
            flat_latent, flat_context = get_flat_latent_and_context(
                (sent_latent2[i]).unsqueeze(0), 5, self.non_zero_pixel_ctx_index
            )
            raw_proba_param = self.arm2(flat_context)
            rate_y, _, _ = compute_rate(flat_latent, raw_proba_param)
            rate_ys.append(rate_y)
            flat_latents.append(flat_latent)
            flat_contexts.append(flat_context)
            raw_proba_params.append(raw_proba_param)
        
        rate_y2 = sum([rate_y.sum() for rate_y in rate_ys])
        # print('sent latent2', rate_y2)

        if get_proba_param:
            

            # Prepare list to accommodate the visualisations
            mus2 = []
            scales2 = []
            latents2 = []

            for i in range(len(sent_latent2)):
                _, flat_mu_y, flat_scale_y = compute_rate(flat_latents[i], raw_proba_params[i])
                # print(flat_mu_y.size(), flat_scale_y.size())
                mu = []
                scale = []
                latent = []

                # "Pointer" for the reading of the 1D scale, mu and rate
                cnt = 0
                h_i, w_i = sent_latent2[i].size()[-2:]
                mu = flat_mu_y.view(-1, h_i, w_i)
                scale = flat_scale_y.view(-1, h_i, w_i)
                latent = sent_latent2[i][0].view(-1, h_i, w_i)

                print(i, mu.size(), scale.size(), latent.size())

                mus2.append(mu)
                scales2.append(scale)
                latents2.append(latent)

            # for i in range(len(sent_latent2)):
            #     _, flat_mu_y, flat_scale_y = compute_rate(flat_latents[i], raw_proba_params[i])
            #     # print(flat_mu_y.size(), flat_scale_y.size())
            #     mu = []
            #     scale = []
            #     latent = []

            #     # "Pointer" for the reading of the 1D scale, mu and rate
            #     cnt = 0
            #     for j, _ in enumerate(sent_latent2[i][0]):
                    
            #         h_i, w_i = sent_latent2[i].size()[-2:]
            #         # print(sent_latent[0][0, i].size())
            #         # print(h_i, w_i, i)

            #         # Scale, mu and rate are 1D tensors where the N latent grids
            #         # are flattened together. As such we have to read the appropriate
            #         # number of values in this 1D vector to reconstruct the i-th grid in 2D
            #         mu_i, scale_i = [
            #             # Read h_i * w_i values starting from cnt
            #             tmp[cnt: cnt + (h_i * w_i)].view((h_i, w_i))
            #             for tmp in [flat_mu_y, flat_scale_y]
            #         ]
            #         # print(i, mu_i.size())

            #         cnt += h_i * w_i
            #         mu.append(mu_i)
            #         scale.append(scale_i)
            #         latent.append(sent_latent2[i][0, j].view(h_i, w_i))
            #     mus2.append(mu)
            #     scales2.append(scale)
            #     latents2.append(latent)

            # print(mus2[0][0].size())
            # print(mus2[1][0].size())

        # sent_latent3
        flat_latents = []
        flat_contexts = []
        raw_proba_params = []
        rate_ys = []
        for i in range(len(sent_latent3)):
            flat_latent, flat_context = get_flat_latent_and_context(
                (sent_latent3[i]).unsqueeze(0), 5, self.non_zero_pixel_ctx_index
            )
            raw_proba_param = self.arm3(flat_context)
            rate_y, _, _ = compute_rate(flat_latent, raw_proba_param)
            rate_ys.append(rate_y)
            flat_latents.append(flat_latent)
            flat_contexts.append(flat_context)
            raw_proba_params.append(raw_proba_param)
        
        rate_y3 = sum([rate_y.sum() for rate_y in rate_ys])
        # print('sent latent3', rate_y3)

        if get_proba_param:
            

            # Prepare list to accommodate the visualisations
            mus3 = []
            scales3 = []
            latents3 = []

            for i in range(len(sent_latent3)):
                _, flat_mu_y, flat_scale_y = compute_rate(flat_latents[i], raw_proba_params[i])
                # print(flat_mu_y.size(), flat_scale_y.size())
                mu = []
                scale = []
                latent = []

                # "Pointer" for the reading of the 1D scale, mu and rate
                cnt = 0
                h_i, w_i = sent_latent3[i].size()[-2:]
                mu = flat_mu_y.view(-1, h_i, w_i)
                scale = flat_scale_y.view(-1, h_i, w_i)
                latent = sent_latent3[i][0].view(-1, h_i, w_i)

                print(i, mu.size(), scale.size(), latent.size())

                mus3.append(mu)
                scales3.append(scale)
                latents3.append(latent)

            # for i in range(len(sent_latent3)):
            #     _, flat_mu_y, flat_scale_y = compute_rate(flat_latents[i], raw_proba_params[i])
            #     # print(flat_mu_y.size(), flat_scale_y.size())
            #     mu = []
            #     scale = []
            #     latent = []

            #     # "Pointer" for the reading of the 1D scale, mu and rate
            #     cnt = 0
            #     # print(sent_latent3[i].size())
            #     # print(sent_latent3[i][0].size())
            #     for j, _ in enumerate(sent_latent3[i][0]):
            #         # print('jjj', j)
                    
            #         h_i, w_i = sent_latent3[i].size()[-2:]
            #         # print(sent_latent[0][0, i].size())
            #         # print(h_i, w_i, i)

            #         # Scale, mu and rate are 1D tensors where the N latent grids
            #         # are flattened together. As such we have to read the appropriate
            #         # number of values in this 1D vector to reconstruct the i-th grid in 2D
            #         mu_i, scale_i = [
            #             # Read h_i * w_i values starting from cnt
            #             tmp[cnt: cnt + (h_i * w_i)].view((h_i, w_i))
            #             for tmp in [flat_mu_y, flat_scale_y]
            #         ]
            #         # print(i, mu_i.size())

            #         cnt += h_i * w_i
            #         mu.append(mu_i)
            #         scale.append(scale_i)
            #         latent.append(sent_latent3[i][0, j].view(h_i, w_i))
            #     mus3.append(mu)
            #     scales3.append(scale)
            #     latents3.append(latent)

            # print(mus3[0][0].size())
            # print(mus3[1][0].size())


            
            
            

        else:
            mu = None
            scale = None
            latent = None

        out: EncoderOutput = {
            'rate_y': rate_y1,
            'mu': mus,
            'scale': scales,
            'latent': latents,
            'rate_y2': rate_y2,
            'mu2': mus2,
            'scale2': scales2,
            'latent2': latents2,
            'rate_y3': rate_y3,
            'mu3': mus3,
            'scale3': scales3,
            'latent3': latents3,
            'total_bits': rate_y1 + rate_y2 + rate_y3
        }
        return out

