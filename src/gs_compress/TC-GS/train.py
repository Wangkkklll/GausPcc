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

import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# # result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')

# os.system('echo $CUDA_VISIBLE_DEVICES')


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
from utils.loss_utils import l1_loss, ssim,wavelet_loss
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm 
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams#SplattingSettings
# from diff_gaussian_rasterization import ExtendedSettings

from utils.encodings import anchor_round_digits, Q_anchor, encoder_anchor, get_binary_vxl_size



from lpipsPyTorch import lpips
torch.hub.set_dir('/hy-tmp/cache')

bit2MB_scale = 8 * 1024 * 1024
run_codec = True

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


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None,splat_args=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,triplane_resolution=dataset.tri_resolution, ste_binary=False)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.updatebbox()
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    torch.cuda.synchronize(); t_start = time.time()
    log_time_sub = 0
    # if not viewpoint_stack:``
    #     viewpoint_stack = scene.getTrainCameras().copy()
    # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

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
        voxel_visible_mask=None
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad,step=iteration,splat_args=splat_args)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]


        bit_per_param = render_pkg["bit_per_param"]
        bit_per_feat_param = render_pkg["bit_per_feat_param"]
        bit_per_scaling_param = render_pkg["bit_per_scaling_param"]
        bit_per_offsets_param = render_pkg["bit_per_offsets_param"]
        lae = render_pkg["autoencoder"]

        if iteration % 2000 == 0 and bit_per_param is not None:

            ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / bit2MB_scale
            ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
            ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[0] * 3 * gaussians.n_offsets / bit2MB_scale
            ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

            logger.info("\n----------------------------------------------------------------------------------------")
            logger.info("\n-----[ITER {}] bits info: bit_per_feat_param={}, anchor_num={}, ttl_size_feat_MB={}-----".format(iteration, bit_per_feat_param.item(), gaussians.get_anchor.shape[0], ttl_size_feat_MB))
            logger.info("\n-----[ITER {}] bits info: bit_per_scaling_param={}, anchor_num={}, ttl_size_scaling_MB={}-----".format(iteration, bit_per_scaling_param.item(), gaussians.get_anchor.shape[0], ttl_size_scaling_MB))
            logger.info("\n-----[ITER {}] bits info: bit_per_offsets_param={}, anchor_num={}, ttl_size_offsets_MB={}-----".format(iteration, bit_per_offsets_param.item(), gaussians.get_anchor.shape[0], ttl_size_offsets_MB))
            logger.info("\n-----[ITER {}] bits info: bit_per_param={}, anchor_num={}, ttl_size_MB={}-----".format(iteration, bit_per_param.item(), gaussians.get_anchor.shape[0], ttl_size_MB))
            logger.info("\n-----[ITER {}] autoencoder info: l1_loss-----{}".format(iteration, lae))
            with torch.no_grad():
                grid_masks = gaussians._mask.data
                binary_grid_masks = (torch.sigmoid(grid_masks) > 0.01).float()
                mask_1_rate, mask_size_bit, mask_size_MB, mask_numel = get_binary_vxl_size(binary_grid_masks + 0.0)  # [0, 1] -> [-1, 1]
            logger.info("\n-----[ITER {}] bits info: 1_rate_mask={}, mask_numel={}, mask_size_MB={}-----".format(iteration, mask_1_rate, mask_numel, mask_size_MB))

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))   
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg  + 0.02 *wavelet_loss(image, gt_image)

        if bit_per_param is not None:
            bit_hash_grid = gaussians.get_encoding_params().numel()
            denom = gaussians._anchor.shape[0]*(gaussians.feat_dim+6+3*gaussians.n_offsets)
            lmbda = 0.001
            loss = loss + lmbda * (bit_per_param + bit_hash_grid / denom)#+ 0.01 *wavelet_loss(image, gt_image)
            if lae is not None:
                loss = loss + lae * 0.01

            loss = loss + 5e-4 * torch.mean(torch.sigmoid(gaussians._mask))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            torch.cuda.synchronize(); t_start_log = time.time()
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),splat_args=splat_args, wandb=wandb, logger=logger, pre_path_name=model_path)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torchvision.utils.save_image(image, os.path.join(model_path, 'render{0:05d}'.format(iteration) + ".png"))
                torchvision.utils.save_image(gt_image, os.path.join(model_path, 'gt{0:05d}'.format(iteration) + ".png"))

            torch.cuda.synchronize(); t_end_log = time.time()
            t_log = t_end_log - t_start_log
            log_time_sub += t_log

            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                # viewspace_point_tensor=screenspace_points: [N_opacity_pos_gaussian, 3]
                # opacity: [N_visible_anchor*K, 1]
                # visibility_filter: radii > 0. 
                # offset_selection_mask: [N_visible_anchor*k]。 
                # voxel_visible_mask:bool = radii_pure > 0
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                if iteration not in range(3000, 4000):  # let the model get fit to quantization
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    torch.cuda.synchronize(); t_end = time.time()
    logger.info("\n Total Training time: {}".format(t_end-t_start-log_time_sub))

    return gaussians.x_bound_max,gaussians.x_bound_min,gaussians.spatial_lr_scale

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

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,splat_args, wandb=None, logger=None,pre_path_name=''):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()

        if 1:
            if iteration == testing_iterations[-1]:
                with torch.no_grad():
                    log_info = scene.gaussians.estimate_final_bits()
                    logger.info(log_info)
                if run_codec:  # conduct encoding and decoding
                    with torch.no_grad():
                        bit_stream_path = os.path.join(pre_path_name, 'bitstreams')
                        os.makedirs(bit_stream_path, exist_ok=True)
                        # conduct encoding
                        patched_infos, log_info = scene.gaussians.conduct_encoding(pre_path_name=bit_stream_path)
                        logger.info(log_info)
                        # conduct decoding
                        log_info = scene.gaussians.conduct_decoding(pre_path_name=bit_stream_path, patched_infos=patched_infos)
                        logger.info(log_info)
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

                    if wandb is not None:
                        gt_image_list = []
                        render_image_list = []
                        errormap_list = []

                    t_list = []

                    for idx, viewpoint in enumerate(config['cameras']):
                        torch.cuda.synchronize(); t_start = time.time()
                        voxel_visible_mask = None#prefilter_voxel(viewpoint, scene.gaussians, *renderArgs,splat_args=splat_args)

                        render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs,splat_args=splat_args, visible_mask=voxel_visible_mask)
                        image = torch.clamp(render_output["render"], 0.0, 1.0)
                        time_sub = render_output["time_sub"]
                        torch.cuda.synchronize(); t_end = time.time()
                        t_list.append(t_end - t_start - time_sub)

                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 30):
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                            if wandb:
                                render_image_list.append(image[None])
                                errormap_list.append((gt_image[None]-image[None]).abs())

                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                                if wandb:
                                    gt_image_list.append(gt_image[None])
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                        lpips_test += lpips(image, gt_image, net_type='vgg').detach().mean().double()

                    psnr_test /= len(config['cameras'])
                    ssim_test /= len(config['cameras'])
                    lpips_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])
                    logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} ssim {} lpips {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                    test_fps = 1.0 / torch.tensor(t_list[0:]).mean()
                    logger.info(f'Test FPS: {test_fps.item():.5f}')
                    if tb_writer:
                        tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
                    if wandb is not None:
                        wandb.log({"test_fps": test_fps, })

                    if tb_writer:
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                        tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    if wandb is not None:
                        wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test}, f"ssim{ssim_test}", f"lpips{lpips_test}")

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background,splat_args=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    psnr_list = []
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = None#prefilter_voxel(view, gaussians, pipeline, background,splat_args=splat_args)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask,splat_args=splat_args)
        torch.cuda.synchronize();t_end = time.time()

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

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, maxcoord,mincoord,skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None,spatial_lr_scale=None,splat_args=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,dataset.tri_resolution,decoded_version=run_codec)
        if mincoord is not None:
            gaussians.x_bound_min = mincoord
            gaussians.x_bound_max = maxcoord
            gaussians.spatial_lr_scale=spatial_lr_scale
            #gaussians.decoded_version=True

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,splat_args=splat_args)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,splat_args=splat_args)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
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
            lpipss.append(lpips(renders[idx], gts[idx]).detach())
        
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
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # ss = SplattingSettings(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=7009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[ 11_000, 15_000,30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 11_000, 15_000,30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    # parser.add_argument("--tri_resolution", type=int, default=256)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
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
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    splat_args = None#ss.get_settings(args)
    # training
    maxcoord,mincoord,spatial_lr_scale = training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger,splat_args=splat_args)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        maxcoord,mincoord,spatial_lr_scale= training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args),maxcoord=maxcoord,mincoord=mincoord,spatial_lr_scale=spatial_lr_scale,wandb=wandb, logger=logger,splat_args=splat_args)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
