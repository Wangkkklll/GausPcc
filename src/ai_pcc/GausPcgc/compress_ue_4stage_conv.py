import os
import time
import random
import argparse

import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torchac

from torchsparse import SparseTensor
from torchsparse.nn import functional as F

from network_ue_4stage_conv import Network

import kit.io as io
import kit.op as op

import pandas as pd

random.seed(1)
np.random.seed(1)
device = 'cuda'

# Set torchsparse configuration
conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)

parser = argparse.ArgumentParser(
    prog='compress_ue_4stage_conv.py',
    description='Compress point cloud geometry data using unequal 4-stage convolution network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', default='./data/kittidet_examples/*.ply', help='Glob pattern for input point cloud files')
parser.add_argument('--output_folder', default='./data/kittidet_compressed/', help='Folder to save compressed bin files')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether input data is pre-quantized")
parser.add_argument('--posQ', default=16, type=int, help='Quantization scale')

parser.add_argument('--channels', type=int, help='Neural network channel count', default=32)
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size', default=3)
parser.add_argument('--ckpt', help='Checkpoint loading path', default='./model/KITTIDetection/ckpt_ue_4stage_conv.pt')

parser.add_argument('--num_samples', default=-1, type=int, help='Randomly select samples for quick testing. [-1 means test all data]')
parser.add_argument('--resultdir', type=str, default='./results', help='Folder to save result CSV files')
parser.add_argument('--prefix', type=str, default='ue_4stage_conv', help='Prefix for result CSV files')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)
os.makedirs(args.resultdir, exist_ok=True)

# Modify file reading method, use sorted to ensure consistent order
file_path_ls = sorted(glob(os.path.join(args.input_glob, '**', '*.*'), recursive=True))
file_path_ls = [f for f in file_path_ls if f.endswith('h5') or f.endswith('ply') or f.endswith('bin') or f.endswith('npy')]

# If sample limit is specified, use the first N instead of random selection
if args.num_samples > 0:
    file_path_ls = file_path_ls[:args.num_samples]

# Use multi-threading to read point clouds
xyz_ls = io.read_point_clouds(file_path_ls)

# Load network
net = Network(channels=args.channels, kernel_size=args.kernel_size)
net.load_state_dict(torch.load(args.ckpt))
net.cuda().eval()

# Warm up
random_coords = torch.randint(low=0, high=2048, size=(2048, 3)).int().cuda()
net(SparseTensor(coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),
                feats=torch.ones((2048, 1))).cuda())

enc_time_ls, bpp_ls, filenames = [], [], []

with torch.no_grad():
    for file_idx in tqdm(range(len(file_path_ls))):
        file_path = file_path_ls[file_idx]
        file_name = os.path.split(file_path)[-1]
        compressed_file_path = os.path.join(args.output_folder, file_name+'.bin')
        
        # Record filename
        filenames.append(file_name)

        ################################ Get xyz coordinates

        if args.is_data_pre_quantized:
            xyz = torch.tensor(xyz_ls[file_idx])
        else:
            xyz = torch.tensor(xyz_ls[file_idx] / 0.001 + 131072)

        xyz = torch.round(xyz / args.posQ).int()
        N = xyz.shape[0]

        xyz = torch.cat((xyz[:,0:1]*0, xyz), dim=-1).int()
        feats = torch.ones((xyz.shape[0], 1), dtype=torch.float)
        x = SparseTensor(coords=xyz, feats=feats).cuda()

        torch.cuda.synchronize()
        enc_time_start = time.time()

        ################################ Preprocessing

        data_ls = []
        while True:
            x = net.fog(x)
            data_ls.append((x.coords.clone(), x.feats.clone())) # must clone
            if x.coords.shape[0] < 64:
                break
        data_ls = data_ls[::-1]

        ################################ Neural network inference

        byte_stream_ls = []
        for depth in range(len(data_ls)-1):
            x_C, x_O = data_ls[depth]
            gt_x_up_C, gt_x_up_O = data_ls[depth+1]
            gt_x_up_C, gt_x_up_O = op.sort_CF(gt_x_up_C, gt_x_up_O)

            # Embedding prior scale features
            x_F = net.prior_embedding(x_O.int()).view(-1, net.channels) # (N_d, C)
            x = SparseTensor(coords=x_C, feats=x_F)
            x = net.prior_resnet(x) # (N_d, C) 

            # Target embedding
            x_up_C, x_up_F = net.fcg(x_C, x_O, x.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)

            x_up_F = net.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = net.target_resnet(x_up)

            # Four-stage unequal bit encoding
            gt_x_up_O_s0 = torch.remainder(torch.div(gt_x_up_O, 128, rounding_mode='floor'), 2)  # 1st bit
            gt_x_up_O_s1 = torch.remainder(torch.div(gt_x_up_O, 64, rounding_mode='floor'), 2)   # 2nd bit
            gt_x_up_O_s2 = torch.remainder(torch.div(gt_x_up_O, 16, rounding_mode='floor'), 4)   # 3rd-4th bits
            gt_x_up_O_s3 = torch.remainder(gt_x_up_O, 16)                                       # 5th-8th bits

            # First stage prediction - 1st bit
            x_up_s0 = net.spatial_conv_s0(x_up)
            x_up_O_prob_s0 = net.pred_head_s0(x_up_s0.feats)
            
            # Second stage prediction - 2nd bit
            s0_emb = net.pred_head_s1_emb(gt_x_up_O_s0[:, 0].long())
            x_up_s1_F = x_up.feats + s0_emb
            x_up_s1 = SparseTensor(coords=x_up.coords, feats=x_up_s1_F)
            x_up_s1 = net.spatial_conv_s1(x_up_s1)
            x_up_O_prob_s1 = net.pred_head_s1(x_up_s1.feats)
            
            # Third stage prediction - 3rd-4th bits
            prev_bits = gt_x_up_O_s0[:, 0] * 2 + gt_x_up_O_s1[:, 0]
            s2_emb = net.pred_head_s2_emb(prev_bits.long())
            x_up_s2_F = x_up.feats + s2_emb
            x_up_s2 = SparseTensor(coords=x_up.coords, feats=x_up_s2_F)
            x_up_s2 = net.spatial_conv_s2(x_up_s2)
            x_up_O_prob_s2 = net.pred_head_s2(x_up_s2.feats)
            
            # Fourth stage prediction - 5th-8th bits
            prev_bits = prev_bits * 4 + gt_x_up_O_s2[:, 0]
            s3_emb = net.pred_head_s3_emb(prev_bits.long())
            x_up_s3_F = x_up.feats + s3_emb
            x_up_s3 = SparseTensor(coords=x_up.coords, feats=x_up_s3_F)
            x_up_s3 = net.spatial_conv_s3(x_up_s3)
            x_up_O_prob_s3 = net.pred_head_s3(x_up_s3.feats)

            # Calculate CDF for each stage and encode separately
            # Stage 1 - processing
            x_up_O_cdf_s0 = torch.cat((x_up_O_prob_s0[:, 0:1]*0, x_up_O_prob_s0.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s0 = torch.clamp(x_up_O_cdf_s0, min=0, max=1)
            x_up_O_cdf_norm_s0 = op._convert_to_int_and_normalize(x_up_O_cdf_s0, True)
            x_up_O_cdf_norm_s0 = x_up_O_cdf_norm_s0.cpu()
            gt_x_up_O_s0_cpu = gt_x_up_O_s0[:, 0].to(torch.int16).cpu()
            
            # Stage 2 - processing
            x_up_O_cdf_s1 = torch.cat((x_up_O_prob_s1[:, 0:1]*0, x_up_O_prob_s1.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s1 = torch.clamp(x_up_O_cdf_s1, min=0, max=1)
            x_up_O_cdf_norm_s1 = op._convert_to_int_and_normalize(x_up_O_cdf_s1, True)
            x_up_O_cdf_norm_s1 = x_up_O_cdf_norm_s1.cpu()
            gt_x_up_O_s1_cpu = gt_x_up_O_s1[:, 0].to(torch.int16).cpu()
            
            # Stage 3 - processing
            x_up_O_cdf_s2 = torch.cat((x_up_O_prob_s2[:, 0:1]*0, x_up_O_prob_s2.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s2 = torch.clamp(x_up_O_cdf_s2, min=0, max=1)
            x_up_O_cdf_norm_s2 = op._convert_to_int_and_normalize(x_up_O_cdf_s2, True)
            x_up_O_cdf_norm_s2 = x_up_O_cdf_norm_s2.cpu()
            gt_x_up_O_s2_cpu = gt_x_up_O_s2[:, 0].to(torch.int16).cpu()
            
            # Stage 4 - processing
            x_up_O_cdf_s3 = torch.cat((x_up_O_prob_s3[:, 0:1]*0, x_up_O_prob_s3.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s3 = torch.clamp(x_up_O_cdf_s3, min=0, max=1)
            x_up_O_cdf_norm_s3 = op._convert_to_int_and_normalize(x_up_O_cdf_s3, True)
            x_up_O_cdf_norm_s3 = x_up_O_cdf_norm_s3.cpu()
            gt_x_up_O_s3_cpu = gt_x_up_O_s3[:, 0].to(torch.int16).cpu()
            
            # Encode all four stages
            byte_stream_s0 = torchac.encode_int16_normalized_cdf(
                x_up_O_cdf_norm_s0, 
                gt_x_up_O_s0_cpu
            )
            
            byte_stream_s1 = torchac.encode_int16_normalized_cdf(
                x_up_O_cdf_norm_s1, 
                gt_x_up_O_s1_cpu
            )
            
            byte_stream_s2 = torchac.encode_int16_normalized_cdf(
                x_up_O_cdf_norm_s2, 
                gt_x_up_O_s2_cpu
            )
            
            byte_stream_s3 = torchac.encode_int16_normalized_cdf(
                x_up_O_cdf_norm_s3, 
                gt_x_up_O_s3_cpu
            )
            
            # Add to total byte stream list
            byte_stream_ls.append(byte_stream_s0)
            byte_stream_ls.append(byte_stream_s1)
            byte_stream_ls.append(byte_stream_s2)
            byte_stream_ls.append(byte_stream_s3)

        # Pack all byte streams into one
        byte_stream = op.pack_byte_stream_ls(byte_stream_ls)

        torch.cuda.synchronize()
        enc_time_end = time.time()

        # Save base point cloud information
        base_x_coords, base_x_feats = data_ls[0]
        base_x_len = base_x_coords.shape[0] 
        base_x_coords = base_x_coords[:, 1:].cpu().numpy() # (n, 3)
        base_x_feats = base_x_feats.cpu().numpy() # (n, 1)

        # Write compressed file
        with open(compressed_file_path, 'wb') as f:
            f.write(np.array(args.posQ, dtype=np.float16).tobytes())
            f.write(np.array(base_x_len, dtype=np.int32).tobytes())
            f.write(np.array(base_x_coords, dtype=np.int32).tobytes())
            f.write(np.array(base_x_feats, dtype=np.uint8).tobytes())
            f.write(byte_stream)
        
        enc_time_ls.append(enc_time_end-enc_time_start)
        bpp_ls.append(op.get_file_size_in_bits(compressed_file_path)/N)
        
        # Save each result to CSV
        results = {
            'filedir': file_name,
            'bpp': bpp_ls[-1],
            'enc_time': enc_time_ls[-1],
            'file_size_bits': op.get_file_size_in_bits(compressed_file_path),
            'num_points': N
        }
        
        results_df = pd.DataFrame([results])
        
        # Initialize or append results
        if file_idx == 0:
            all_results = results_df.copy(deep=True)
        else:
            all_results = pd.concat([all_results, results_df], ignore_index=True)
        
        # Save current results
        csvfile = os.path.join(args.resultdir, args.prefix + '_data' + str(len(file_path_ls)) + '.csv')
        all_results.to_csv(csvfile, index=False)

# Calculate averages
average_results = all_results.mean(numeric_only=True).to_dict()
average_results['filedir'] = 'avg'  # Replace filename with "avg"

# Convert average results to DataFrame and add to results
average_df = pd.DataFrame([average_results])
all_results_with_avg = pd.concat([all_results, average_df], ignore_index=True)

# Save final results (including averages)
csvfile = os.path.join(args.resultdir, args.prefix + '_data' + str(len(file_path_ls)) + '.csv')
all_results_with_avg.to_csv(csvfile, index=False)

print('Total: {total_n:d} | Average bitrate:{bpp:.3f} | Encoding time:{enc_time:.3f}s | Max GPU memory:{memory:.2f}MB'.format(
    total_n=len(enc_time_ls),
    bpp=np.array(bpp_ls).mean(),
    enc_time=np.array(enc_time_ls).mean(),
    memory=torch.cuda.max_memory_allocated()/1024/1024
))
print('Results saved to ', csvfile)
print(all_results.mean(numeric_only=True))