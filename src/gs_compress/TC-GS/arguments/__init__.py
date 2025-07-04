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

from argparse import ArgumentParser, Namespace
import json
# from diff_gaussian_rasterization import ExtendedSettings, GlobalSortOrder, SortMode
from distutils.util import strtobool

import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.tri_resolution = 256
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0


        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        
        self.feature_lr_init = 0.005#triplane
        self.feature_lr_final = 0.00001
        self.feature_lr_delay_mult = 0.01
        self.feature_lr_max_steps = 30_000

        self.mask_lr_init = 0.01
        self.mask_lr_final = 0.0001
        self.mask_lr_delay_mult = 0.01
        self.mask_lr_max_steps = 30_000
        
        self.mlp_tri_lr_init = 0.005
        self.mlp_tri_lr_final = 0.00001
        self.mlp_tri_lr_delay_mult = 0.01
        self.mlp_tri_lr_max_steps = 30_000
        
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.0001
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        super().__init__(parser, "Optimization Parameters")

# class SplattingSettings():
    
#     group_config = None
#     group_settings = None
#     settings = ExtendedSettings()
#     parser = None
#     render = False
    
#     def __init__(self, parser, render=False):
#         self.parser = parser
#         self.render = render
#         if not render:
#             self.group_config = parser.add_argument_group("Splatting Config")
#             self.group_config.add_argument("--splatting_config", type=str)
            
#         bool_ = lambda x: bool(strtobool(x))
        
#         self.group_settings = parser.add_argument_group("Splatting Settings")
#         self.group_settings.add_argument("--sort_mode", type=lambda sortmode: SortMode[sortmode], choices=list(SortMode))
#         self.group_settings.add_argument("--sort_order", type=lambda sortorder: GlobalSortOrder[sortorder], choices=list(GlobalSortOrder))
#         self.group_settings.add_argument("--tile_4x4", type=int, choices=[64], help='only needed if using sort_mode HIER')
#         self.group_settings.add_argument("--tile_2x2", type=int, choices=[8,12,20], help='only needed if using sort_mode HIER')
#         self.group_settings.add_argument("--per_pixel", type=int, choices=[1,2,4,8,12,16,20,24], help='if using sort_mode HIER, only {4,8,16} are valid')
#         self.group_settings.add_argument("--rect_bounding", type=bool_, choices=[True, False], help="Bound 2D Gaussians with a rectangle instead of a circle")
#         self.group_settings.add_argument("--tight_opacity_bounding", type=bool_, choices=[True, False], help="Bound 2D Gaussians by considering their opacity")
#         self.group_settings.add_argument("--tile_based_culling", type=bool_, choices=[True, False], help="Cull complete tiles based on opacity")
#         self.group_settings.add_argument("--hierarchical_4x4_culling", type=bool_, choices=[True, False], help="Cull Gaussians for 4x4 subtiles, only when using sort_mode HIER")
#         self.group_settings.add_argument("--load_balancing", type=bool_, choices=[True, False], help="Perform per-tile computations cooperatively (e.g. duplication)")
#         self.group_settings.add_argument("--proper_ewa_scaling", type=bool_, choices=[True, False], help='Dilation of 2D Gaussians as proposed by Yu et al. ("Mip-Splatting")')
    
#     def get_settings(self, arguments):
#         # get valid choices from configargparse
#         config = None
        
#         # load default dict, if passed
#         if self.render:
#             cmdlne_string = sys.argv[1:]
#             args_cmdline = self.parser.parse_args(cmdlne_string)
#             cfgfilepath = os.path.join(args_cmdline.model_path, "config.json")
#             print("Looking for splatting config file in", cfgfilepath)
#             if os.path.exists(cfgfilepath):
#                 print("Config file found: {}".format(cfgfilepath))
#                 self.settings = ExtendedSettings.from_json(cfgfilepath)
#             else:
#                 print("No config file found, assuming default values")
#         else:
#             for arg in vars(arguments).items():
#                 if any([arg[0] in z.option_strings[0] for z in self.group_config._group_actions]):
#                     # json passed, load it
#                     if arg[1] is None:
#                         continue
#                     with open(arg[1], 'r') as json_file:
#                         config = json.load(json_file)
#                         self.settings = ExtendedSettings.from_dict(config)
                    
#         for arg in vars(arguments).items():
#             if any([arg[0] in z.option_strings[0] for z in self.group_settings._group_actions]):
#                 # pass any options which were not given
#                 if arg[1] is None:
#                     continue
#                 self.settings.set_value(arg[0], arg[1])
                
#         return self.settings

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
