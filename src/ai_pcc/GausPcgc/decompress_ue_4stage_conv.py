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

random.seed(1)
np.random.seed(1)
device = 'cuda'

# Set torchsparse configuration
conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)

parser = argparse.ArgumentParser(
    prog='decompress_ue_4stage_conv.py',
    description='Decompress point cloud geometry data using unequal 4-stage convolution network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', default='./data/kittidet_compressed/*.bin', help='Glob pattern for input compressed files')
parser.add_argument('--output_folder', default='./data/kittidet_decompressed/', help='Folder to save decompressed PLY files')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether original point cloud is pre-quantized")

parser.add_argument('--channels', type=int, help='Neural network channel count', default=32)
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size', default=3)
parser.add_argument('--ckpt', help='Checkpoint loading path', default='./model/KITTIDetection/ckpt_ue_4stage_conv.pt')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

file_path_ls = glob(args.input_glob)

# Load network
net = Network(channels=args.channels, kernel_size=args.kernel_size)
net.load_state_dict(torch.load(args.ckpt))
net.cuda().eval()

# Warm up
random_coords = torch.randint(low=0, high=2048, size=(2048, 3)).int().cuda()
net(SparseTensor(coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),
                feats=torch.ones((2048, 1))).cuda())

dec_time_ls = []

with torch.no_grad():
    for file_path in tqdm(file_path_ls):
        file_name = os.path.split(file_path)[-1]
        decompressed_file_path = os.path.join(args.output_folder, file_name+'.ply')

        ################################ Read bin file

        with open(file_path, 'rb') as f:
            posQ = np.frombuffer(f.read(2), dtype=np.float16)[0]
            base_x_len = np.frombuffer(f.read(4), dtype=np.int32)[0]
            base_x_coords = np.frombuffer(f.read(base_x_len*4*3), dtype=np.int32)
            base_x_feats = np.frombuffer(f.read(base_x_len*1), dtype=np.uint8)
            byte_stream = f.read()

        torch.cuda.synchronize()
        dec_time_start = time.time()

        ################################ Decompression

        base_x_coords = torch.tensor(base_x_coords.reshape(-1, 3), device=device) 
        base_x_feats = torch.tensor(base_x_feats.reshape(-1, 1), device=device)

        # Initialize x as base point cloud
        x = SparseTensor(coords=torch.cat((base_x_feats*0, base_x_coords), dim=-1), feats=base_x_feats).cuda()
        # Unpack byte stream
        byte_stream_ls = op.unpack_byte_stream(byte_stream)

        for byte_stream_idx in range(0, len(byte_stream_ls), 4):  # Each loop processes 4 stages of byte streams
            byte_stream_s0 = byte_stream_ls[byte_stream_idx]     # Byte stream for 1st bit
            byte_stream_s1 = byte_stream_ls[byte_stream_idx+1]   # Byte stream for 2nd bit
            byte_stream_s2 = byte_stream_ls[byte_stream_idx+2]   # Byte stream for 3rd-4th bits
            byte_stream_s3 = byte_stream_ls[byte_stream_idx+3]   # Byte stream for 5th-8th bits

            # embedding prior scale feats
            x_O = x.feats.int()
            x_F = net.prior_embedding(x_O).view(-1, net.channels) # (N_d, C)
            x = SparseTensor(coords=x.coords, feats=x_F)
            x = net.prior_resnet(x) # (N_d, C)

            # target embedding
            x_up_C, x_up_F = net.fcg(x.coords, x_O, x_F=x.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)

            x_up_F = net.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = net.target_resnet(x_up)

            # First stage decoding - 1st bit, using spatial convolution
            x_up_s0 = net.spatial_conv_s0(x_up)
            x_up_O_prob_s0 = net.pred_head_s0(x_up_s0.feats)  # (N, 2)
            x_up_O_cdf_s0 = torch.cat((x_up_O_prob_s0[:, 0:1]*0, x_up_O_prob_s0.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s0 = torch.clamp(x_up_O_cdf_s0, min=0, max=1)
            x_up_O_cdf_s0_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s0, True)
            x_up_O_cdf_s0_norm = x_up_O_cdf_s0_norm.cpu()
            
            # Decode 1st bit
            x_up_O_s0 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s0_norm, byte_stream_s0).cuda()

            # Second stage decoding - 2nd bit, based on first stage result
            s0_emb = net.pred_head_s1_emb(x_up_O_s0.long())
            x_up_s1_F = x_up.feats + s0_emb
            x_up_s1 = SparseTensor(coords=x_up.coords, feats=x_up_s1_F)
            x_up_s1 = net.spatial_conv_s1(x_up_s1)
            x_up_O_prob_s1 = net.pred_head_s1(x_up_s1.feats)  # (N, 2)
            x_up_O_cdf_s1 = torch.cat((x_up_O_prob_s1[:, 0:1]*0, x_up_O_prob_s1.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s1 = torch.clamp(x_up_O_cdf_s1, min=0, max=1)
            x_up_O_cdf_s1_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s1, True)
            x_up_O_cdf_s1_norm = x_up_O_cdf_s1_norm.cpu()
            
            # Decode 2nd bit
            x_up_O_s1 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s1_norm, byte_stream_s1).cuda()

            # Third stage decoding - 3rd-4th bits, based on previous two stages
            prev_bits = x_up_O_s0 * 2 + x_up_O_s1  # Combine first 2 bits
            s2_emb = net.pred_head_s2_emb(prev_bits.long())
            x_up_s2_F = x_up.feats + s2_emb
            x_up_s2 = SparseTensor(coords=x_up.coords, feats=x_up_s2_F)
            x_up_s2 = net.spatial_conv_s2(x_up_s2)
            x_up_O_prob_s2 = net.pred_head_s2(x_up_s2.feats)  # (N, 4)
            x_up_O_cdf_s2 = torch.cat((x_up_O_prob_s2[:, 0:1]*0, x_up_O_prob_s2.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s2 = torch.clamp(x_up_O_cdf_s2, min=0, max=1)
            x_up_O_cdf_s2_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s2, True)
            x_up_O_cdf_s2_norm = x_up_O_cdf_s2_norm.cpu()
            
            # Decode 3rd-4th bits
            x_up_O_s2 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s2_norm, byte_stream_s2).cuda()

            # Fourth stage decoding - 5th-8th bits, based on previous three stages
            prev_bits = prev_bits * 4 + x_up_O_s2  # Combine first 4 bits
            s3_emb = net.pred_head_s3_emb(prev_bits.long())
            x_up_s3_F = x_up.feats + s3_emb
            x_up_s3 = SparseTensor(coords=x_up.coords, feats=x_up_s3_F)
            x_up_s3 = net.spatial_conv_s3(x_up_s3)
            x_up_O_prob_s3 = net.pred_head_s3(x_up_s3.feats)  # (N, 16)
            x_up_O_cdf_s3 = torch.cat((x_up_O_prob_s3[:, 0:1]*0, x_up_O_prob_s3.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s3 = torch.clamp(x_up_O_cdf_s3, min=0, max=1)
            x_up_O_cdf_s3_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s3, True)
            x_up_O_cdf_s3_norm = x_up_O_cdf_s3_norm.cpu()
            
            # Decode 5th-8th bits
            x_up_O_s3 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s3_norm, byte_stream_s3).cuda()

            # Combine all bits to restore complete 8-bit value
            x_up_O = (x_up_O_s0 * 128 + x_up_O_s1 * 64 + x_up_O_s2 * 16 + x_up_O_s3).unsqueeze(-1)
            
            # Update current layer's sparse tensor
            x = SparseTensor(coords=x_up_C, feats=x_up_O).cuda()
            
        # Call fcg to get the final point cloud
        scan = net.fcg(x.coords, x.feats.int())
        
        # Print point count information
        print(f"Points after decompression: {scan.shape[0]}")
        
        if args.is_data_pre_quantized:
            scan = scan[:, 1:] * posQ
        else:
            scan = (scan[:, 1:] * posQ - 131072) * 0.001

        torch.cuda.synchronize()
        dec_time_end = time.time()

        dec_time_ls.append(dec_time_end-dec_time_start)

        io.save_ply_ascii_geo(scan.float().cpu().numpy(), decompressed_file_path)

print('Total: {total_n:d} | Decoding time:{dec_time:.3f}s | Max GPU memory:{memory:.2f}MB'.format(
    total_n=len(dec_time_ls),
    dec_time=np.array(dec_time_ls).mean(),
    memory=torch.cuda.max_memory_allocated()/1024/1024
))