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
import numpy as np
from comet_ml import Experiment, ExistingExperiment
import subprocess

import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
# import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.encodings import anchor_round_digits, Q_anchor, encoder_anchor, get_binary_vxl_size
from lpipsPyTorch import lpips

bit2MB_scale = 8 * 1024 * 1024
run_codec = True
save_train_images = False
save_test_images = True

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print('Backup Finished!')


def training(args_param, dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    attribute_cfg = dataset.attribute_config
    
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        n_features_per_level=args_param.n_features,
        chcm_slices_list=args_param.chcm_slices_list,
        chcm_for_offsets=args_param.chcm_for_offsets,
        chcm_for_scaling=args_param.chcm_for_scaling,
        attribute_config=dataset.attribute_config
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path)


    

    # gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        scene.load_model(first_iter)
        gaussians.restore(model_params, opt)
    else:
        gaussians.training_setup(opt)
    gaussians.update_anchor_bound()
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ncols=100)
    first_iter += 1
    torch.cuda.synchronize(); t_start = time.time()
    log_time_sub = 0
    mask_weight = torch.zeros(5, dtype=torch.float, device="cuda")
    
    for iteration in range(first_iter, opt.iterations + 1):
        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        ##################        
        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if iteration >= 0:
            if mask_weight.shape[0] != gaussians.get_anchor.shape[0] or iteration == 0: #重算mask
                tmp_viewpoint_stack = scene.getTrainCameras().copy()
                anchor_visible_mask = torch.zeros(gaussians.get_anchor.shape[0], dtype=torch.float, device="cuda")
                n = len(tmp_viewpoint_stack)
            
                for ii in range(n):
                    viewpoint_cam = tmp_viewpoint_stack.pop(0)
                    
                    voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
                    anchor_visible_mask += voxel_visible_mask.to(torch.float)
                anchor_visible_mask = anchor_visible_mask.view(-1, 1)
                k = args_param.cam_mask
                p = anchor_visible_mask / torch.mean(anchor_visible_mask)
                # the smaller the mask is, num of anchor is smaller
                p = p.repeat(1, 10).unsqueeze(-1)
                mask_weight = k * p


        if iteration == opt.triplane_init_fit_iter:
            gaussians.set_feature_net(p[:, 0].view(-1))
            gaussians.training_setup_triplane(opt)
        

        

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background, step=iteration)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration, mask=mask_weight)
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity= render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        
        feat_rate_per_param = render_pkg["feat_rate_per_param"]
        bit_per_param = render_pkg["bit_per_param"]
        bit_per_feat_param = render_pkg["bit_per_feat_param"]
        bit_per_scaling_param = render_pkg["bit_per_scaling_param"]
        bit_per_offsets_param = render_pkg["bit_per_offsets_param"]
        ####only consider offset and offset
        if iteration % 2000 == 0 and bit_per_param is not None:
            ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / bit2MB_scale
            ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
            ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[0] * 3 * gaussians.n_offsets / bit2MB_scale
            ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

            with torch.no_grad():
                grid_masks = gaussians._mask.data
                binary_grid_masks = (torch.sigmoid(grid_masks) > 0.01).float()
                mask_1_rate, mask_size_bit, mask_size_MB, mask_numel = get_binary_vxl_size(binary_grid_masks + 0.0)  # [0, 1] -> [-1, 1]
            a= "train"
            b1="scaling"
            b2="offset"
            b3="feat"
            log = {
                    f'{a} {b1} bit_per_param': bit_per_scaling_param.item(),
                    f'{a} {b1} ttl_size_MB': ttl_size_scaling_MB,
                    f'{a} {b2} bit_per_param': bit_per_offsets_param.item(),
                    f'{a} {b2} ttl_size_MB': ttl_size_offsets_MB,
                    f'{a} {b3} bit_per_param':bit_per_feat_param,
                    f'{a} {b1}+{b2}+{b3} bit_per_param': bit_per_param.item(),
                    f'{a} {b1}+{b2}+{b3} ttl_size_MB': ttl_size_MB,
                    f'{a} feature size(MB)':  feat_rate_per_param / bit2MB_scale   
                }
            logger.log_metrics(log, step=iteration)
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        scaffold_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
        mask_loss = torch.mean(torch.sigmoid(gaussians._mask))
        
        if iteration>=3000:
            loss = scaffold_loss +  max(1e-3, 0.3 * args_param.lmbda) * mask_loss
        else:
            loss = scaffold_loss
            
        denom = gaussians._anchor.shape[0]*(gaussians.feat_dim+6+3*gaussians.n_offsets)
        if opt.triplane_init_fit_iter + 5000 < iteration < opt.triplane_init_fit_iter + 6000:
            loss = feat_rate_per_param
        elif opt.triplane_init_fit_iter <= iteration and bit_per_param is not None:
            loss = loss + args_param.lmbda * (args_param.lmbda_tri * feat_rate_per_param / denom + bit_per_param)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            F_bits =0
            if iteration > opt.triplane_init_fit_iter:
                F_bits = feat_rate_per_param / bit2MB_scale
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "psnr": f"{psnr(gt_image, image).mean().double():.{2}f}","F_MB": f"{F_bits:.{2}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            torch.cuda.synchronize(); t_start_log = time.time()
            training_report(dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger, args_param.model_path, mask_weight, args_param)
            if (iteration in saving_iterations):
                scene.save(iteration)
            torch.cuda.synchronize(); t_end_log = time.time()
            t_log = t_end_log - t_start_log
            log_time_sub += t_log

            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                if iteration not in range(3000, 4000):  # let the model get fit to quantization##############no need
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                
            if iteration < opt.iterations:
                if iteration < opt.triplane_init_fit_iter:
                    gaussians.optimizer.step()
                elif iteration >= opt.triplane_init_fit_iter and iteration < opt.triplane_init_fit_iter + 5000:
                    if iteration == opt.triplane_init_fit_iter:
                        set_requires_grad(gaussians.feature_net,True)
                        set_requires_grad(gaussians.feature_net.attribute_net.grid.arm,False)
                        set_requires_grad(gaussians.feature_net.attribute_net.grid.arm2,False)
                        set_requires_grad(gaussians.feature_net.attribute_net.grid.arm3,False)
                        set_requires_grad(gaussians.mlp_context_from_f1, True)
                    gaussians.optimizer.step()
                    gaussians.feature_net_optimizer.step()
                    gaussians.feature_grid_optimizer.step()
                elif iteration >= opt.triplane_init_fit_iter + 5000 and iteration < opt.triplane_init_fit_iter + 6000:
                    if iteration ==opt.triplane_init_fit_iter + 6000:
                        set_requires_grad(gaussians.feature_net,False)
                        set_requires_grad(gaussians.feature_net.attribute_net.grid.arm,True)
                        set_requires_grad(gaussians.feature_net.attribute_net.grid.arm2,True)
                        set_requires_grad(gaussians.feature_net.attribute_net.grid.arm3,True)
                        set_requires_grad(gaussians.mlp_context_from_f1, False)
                        for i in gaussians.l_param:
                            set_requires_grad(i,False)
                    gaussians.feature_arm_optimizer.step()
                elif iteration >= opt.triplane_init_fit_iter + 6000 and iteration < opt.triplane_init_fit_iter + 9000:
                    if iteration ==opt.triplane_init_fit_iter + 6000:
                        set_requires_grad(gaussians.feature_net,True)
                        set_requires_grad(gaussians.feature_net.attribute_net.grid.grids,False)
                        set_requires_grad(gaussians.mlp_context_from_f1, True)
                        for i in gaussians.l_param:
                            set_requires_grad(i,True)
                    gaussians.feature_arm_optimizer.step()
                    gaussians.feature_net_optimizer.step()
                    gaussians.optimizer.step()
                elif iteration >= opt.triplane_init_fit_iter + 9000 and iteration < opt.triplane_init_fit_iter + 35000:
                    if iteration ==opt.triplane_init_fit_iter + 9000:
                        set_requires_grad(gaussians.feature_net.attribute_net.grid.grids,True)
                    gaussians.optimizer.step()
                    gaussians.feature_arm_optimizer.step()
                    gaussians.feature_net_optimizer.step()
                    gaussians.feature_grid_optimizer.step()

                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration >= opt.triplane_init_fit_iter:
                    gaussians.feature_net_optimizer.zero_grad(set_to_none = True)
                    gaussians.feature_arm_optimizer.zero_grad(set_to_none = True)
                    gaussians.feature_grid_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    torch.cuda.synchronize(); t_end = time.time()


    return gaussians.x_bound_min, gaussians.x_bound_max

def set_requires_grad(module_or_param, value):
    if isinstance(module_or_param, torch.nn.Module):
        for param in module_or_param.parameters():
            if param.dtype.is_floating_point or param.dtype.is_complex:
                param.requires_grad = value
    elif isinstance(module_or_param, torch.nn.Parameter):
        if module_or_param.dtype.is_floating_point or module_or_param.dtype.is_complex:
            module_or_param.requires_grad = value
    else:
        raise TypeError("Input must be an nn.Module or nn.Parameter")
        
def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, pre_path_name='', weighted_mask=None, args_param=None):
   
    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        rate_set=None
        if 1:
            if iteration == testing_iterations[-1]:
                with torch.no_grad():
                    log_info = scene.gaussians.estimate_final_bits(weighted_mask)
                    print(log_info)
            if iteration == testing_iterations[-1]:
                if run_codec:  # conduct encoding and decoding
                    with torch.no_grad():
                        bit_stream_path = os.path.join(pre_path_name, 'bitstreams')
                        os.makedirs(bit_stream_path, exist_ok=True)
                        # conduct encoding
                        patched_infos, log_info, rate_set = scene.gaussians.conduct_encoding(pre_path_name=bit_stream_path, weighted_mask=weighted_mask)
                        print(log_info)
                        # conduct decoding
                        log_info = scene.gaussians.conduct_decoding(pre_path_name=bit_stream_path, patched_infos=patched_infos, weighted_mask=weighted_mask)
                        weighted_mask = None
                        
                        print(log_info)
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                                  {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

            for config in validation_configs:
                # if config['name'] == 'test': assert len(config['cameras']) == 200
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    ssim_test = 0.0
                    lpips_test = 0.0

                    
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                    t_list = []

                    for idx, viewpoint in enumerate(config['cameras']):
                        torch.cuda.synchronize(); t_start = time.time()
                        voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs, step=iteration)
                        # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                        render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask, step=iteration, mask=weighted_mask)
                        image = torch.clamp(render_output["render"], 0.0, 1.0)
                        time_sub = render_output["time_sub"]
                        torch.cuda.synchronize(); t_end = time.time()
                        t_list.append(t_end - t_start - time_sub)

                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                        gt_image_list.append(gt_image)
                        render_image_list.append(image)
                        errormap_list.append((image - gt_image).abs())
                        
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                        # lpips_test += lpips_fn(image, gt_image, normalize=True).detach().mean().double()
                        lpips_test += lpips(image, gt_image, net_type='vgg').detach().mean().double()

                    psnr_test /= len(config['cameras'])
                    ssim_test /= len(config['cameras'])
                    lpips_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])

                    
                    
                    if rate_set != None:
                        tot = sum(rate_set) - rate_set[1]
                        
                    if (config['name'] == 'test' and save_test_images) or (config['name'] == 'train' and save_train_images):
                        for image_list, name in zip([gt_image_list, render_image_list, errormap_list], ["gt", "ours", "error"]):
                            os.makedirs(os.path.join(args_param.model_path, f"iteration_{iteration}/{config['name']}/{name}"), exist_ok=True)
                            for idx, img in enumerate(image_list):
                                torchvision.utils.save_image(img, os.path.join(args_param.model_path, f"iteration_{iteration}/{config['name']}/{name}/{idx:05d}.png"))
                    
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} ssim {} lpips {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                    test_fps = 1.0 / torch.tensor(t_list[0:]).mean()
                    print(f'Test FPS: {test_fps.item():.5f}')
                    if args_param.txt_name is not None and iteration == testing_iterations[-1] and config['name'] == 'test' and rate_set != None:
                        write_to_file(args_param.txt_name, args_param.lmbda ,tot, psnr_test, ssim_test, lpips_test, rate_set)
                    
        torch.cuda.empty_cache()

        scene.gaussians.train()
def write_to_file(txt_name, lmbda, size, psnr, ssim, lpips, rate_set):
    # Check if the txt_name is provided
    if txt_name is not None:
        file_name = f"{txt_name}.txt"

        folder = os.path.dirname(file_name)

        # 如果資料夾不存在，建立資料夾
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Define the column headers and format
        headers = ["lmbda", "size", "psnr", "ssim", "lpips", "anchor", "total feat"] + [f'feat{i}' for i in range(len(rate_set)-7)] + ["scaling", "offsets", "masks", "MLPs", "triplane_f"]
        header_format = "{:<10} {:<10} {:<10} {:<10} {:<10} " + " ".join(["{:<10}"] * len(rate_set)) + "\n"
        row_format = "{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} " + " ".join(["{:<10.4f}"] * len(rate_set)) + "\n"
        # Check if the file does not exist, and establish it with the given column names
        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                # Write the headers with alignment
                f.write(header_format.format(*headers))
        
        # Write the data row with alignment
        with open(file_name, 'a') as f:
            f.write(row_format.format(lmbda, size, psnr, ssim, lpips, *rate_set))

        print(f"Data written to {file_name}")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    psnr_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t_start = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, step=iteration)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]

        #
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        render_image = torch.clamp(rendering.to("cuda"), 0.0, 1.0)
        psnr_view = psnr(render_image, gt_image).mean().double()
        psnr_list.append(psnr_view)

        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    print('testing_float_psnr=:', sum(psnr_list) / len(psnr_list))

    return t_list, visible_count_list


def render_sets(args_param, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, x_bound_min=None, x_bound_max=None):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            n_features_per_level=args_param.n_features,
            chcm_slices_list=args_param.chcm_slices_list,
            chcm_for_offsets=args_param.chcm_for_offsets,
            chcm_for_scaling=args_param.chcm_for_scaling,
            attribute_config=dataset.attribute_config
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()
        if x_bound_min is not None:
            gaussians.x_bound_min = x_bound_min
            gaussians.x_bound_max = x_bound_max

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            t_train_list, _  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            print(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            print(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })

    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)

            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)

        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[24999, 25000, 30000, 40000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[40000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--no_comet", action="store_true")
    parser.add_argument("--project_name", type=str, default = 'S3DGS')
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--log2", type=int, default = 13)
    parser.add_argument("--log2_2D", type=int, default = 15)
    parser.add_argument("--n_features", type=int, default = 4)
    parser.add_argument("--chcm_slices_list", nargs="+", type=int, default=[12, 12, 13, 13])
    parser.add_argument("--chcm_for_offsets", action='store_true', default=False)
    parser.add_argument("--chcm_for_scaling", action='store_true', default=False)
    parser.add_argument("--lmbda", type=float)
    parser.add_argument("--lmbda_tri", type=float, default=10.0)
    parser.add_argument('--exp_name',  type=str, required=True)
    parser.add_argument('--exp_key',   type=str, default=None)
    parser.add_argument('--cam_mask', type=float, default=1)
    parser.add_argument('--txt_name', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)

    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations = args.checkpoint_iterations 
    args.save_iterations.append(args.iterations)

    if args.config is not None:
        def merge_hparams(args, config):
            params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
            for param in params:
                if param in config.keys():
                    for key, value in config[param].items():
                        if hasattr(args, key):
                            setattr(args, key, value)

            return args
        import mmengine
        config = mmengine.Config.fromfile(args.config)
        args = merge_hparams(args, config)



    # enable logging
    import json
    with open("./comet_keys.json", 'r') as f:
        personal = json.load(f)
    comet_logger = Experiment(
        api_key=personal['api_key'],
        workspace=personal['workspace'],
        project_name=args.project_name,
        disabled=args.no_comet
    )

    comet_logger.set_name(args.exp_name)
    comet_logger.log_parameters(args)
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)
    
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    args.port = np.random.randint(10000, 20000)
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    x_bound_min, x_bound_max = training(args, lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger=comet_logger)
    
    print("\nTraining complete.")


