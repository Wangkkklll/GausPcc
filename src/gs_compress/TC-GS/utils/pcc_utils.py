import os,sys
import time
import torch
import torchac
import numpy as np
from torchsparse import SparseTensor
from torchsparse.nn import functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '../../GausPcgc'))
import kit.op as op


def calculate_morton_order(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate Morton order of the input points.
    """
    assert len(x.shape) == 2 and x.shape[1] == 3, f'Input data must be a 3D point cloud, but got {x.shape}.'
    device = x.device
    x = x - torch.min(x, dim=0, keepdim=True)[0]
    x = x.cpu().numpy().astype(np.int64)
    indices_sorted = np.argsort(x @ np.power(x.max() + 1, np.arange(x.shape[1])), axis=0)
    indices_sorted = torch.tensor(indices_sorted, dtype=torch.long, device=device)
    return indices_sorted

def compress_point_cloud(
    xyz_quantized,            # Quantized point cloud coordinates, numpy array or torch tensor
    ckpt_path,                # Path to pre-trained weights file
    output_path,              # Output bin file path
    channels=32,              # Network channel count
    kernel_size=5,            # Convolution kernel size
    posQ=1                   # Quantization scale
):
    """
    Compress point cloud into a bin file
    
    Parameters:
        xyz_quantized: Quantized point cloud coordinates (N, 3) as numpy array or torch tensor
        ckpt_path: Path to pre-trained model weights file
        output_path: Path for output bin file
        channels: Neural network channel count
        kernel_size: Convolution kernel size
        posQ: Quantization scale
    
    Returns:
        dict: Dictionary containing compression result info (bitrate, encoding time, file size, etc.)
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set torchsparse configuration
    conv_config = F.conv_config.get_default_conv_config()
    conv_config.kmap_mode = "hashmap"
    F.conv_config.set_global_conv_config(conv_config)
    
    # Import network model (needs to be imported in the file using this function)
    from network_ue_4stage_conv import Network
    
    # Convert input to torch tensor
    if isinstance(xyz_quantized, np.ndarray):
        xyz = torch.tensor(xyz_quantized)
    else:
        xyz = xyz_quantized.clone()
        
    # Load network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Network(channels=channels, kernel_size=kernel_size)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.to(device).eval()
    
    # Record number of points
    N = xyz.shape[0]
    
    # Build input tensor
    xyz = torch.cat((xyz[:,0:1]*0, xyz), dim=-1).int()
    feats = torch.ones((xyz.shape[0], 1), dtype=torch.float)
    x = SparseTensor(coords=xyz, feats=feats).to(device)
    
    # Start encoding
    torch.cuda.synchronize() if device == 'cuda' else None
    enc_time_start = time.time()
    
    with torch.no_grad():
        ################################ Preprocessing
        data_ls = []
        while True:
            x = net.fog(x)
            data_ls.append((x.coords.clone(), x.feats.clone()))
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
            x_F = net.prior_embedding(x_O.int()).view(-1, net.channels)
            x = SparseTensor(coords=x_C, feats=x_F)
            x = net.prior_resnet(x)
            
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
            gt_x_up_O_s3 = torch.remainder(gt_x_up_O, 16)                                        # 5th-8th bits
            
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
            
            # Encode data for each stage
            # Stage 1
            x_up_O_cdf_s0 = torch.cat((x_up_O_prob_s0[:, 0:1]*0, x_up_O_prob_s0.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s0 = torch.clamp(x_up_O_cdf_s0, min=0, max=1)
            x_up_O_cdf_norm_s0 = op._convert_to_int_and_normalize(x_up_O_cdf_s0, True)
            x_up_O_cdf_norm_s0 = x_up_O_cdf_norm_s0.cpu()
            gt_x_up_O_s0_cpu = gt_x_up_O_s0[:, 0].to(torch.int16).cpu()
            
            # Stage 2
            x_up_O_cdf_s1 = torch.cat((x_up_O_prob_s1[:, 0:1]*0, x_up_O_prob_s1.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s1 = torch.clamp(x_up_O_cdf_s1, min=0, max=1)
            x_up_O_cdf_norm_s1 = op._convert_to_int_and_normalize(x_up_O_cdf_s1, True)
            x_up_O_cdf_norm_s1 = x_up_O_cdf_norm_s1.cpu()
            gt_x_up_O_s1_cpu = gt_x_up_O_s1[:, 0].to(torch.int16).cpu()
            
            # Stage 3
            x_up_O_cdf_s2 = torch.cat((x_up_O_prob_s2[:, 0:1]*0, x_up_O_prob_s2.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s2 = torch.clamp(x_up_O_cdf_s2, min=0, max=1)
            x_up_O_cdf_norm_s2 = op._convert_to_int_and_normalize(x_up_O_cdf_s2, True)
            x_up_O_cdf_norm_s2 = x_up_O_cdf_norm_s2.cpu()
            gt_x_up_O_s2_cpu = gt_x_up_O_s2[:, 0].to(torch.int16).cpu()
            
            # Stage 4
            x_up_O_cdf_s3 = torch.cat((x_up_O_prob_s3[:, 0:1]*0, x_up_O_prob_s3.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s3 = torch.clamp(x_up_O_cdf_s3, min=0, max=1)
            x_up_O_cdf_norm_s3 = op._convert_to_int_and_normalize(x_up_O_cdf_s3, True)
            x_up_O_cdf_norm_s3 = x_up_O_cdf_norm_s3.cpu()
            gt_x_up_O_s3_cpu = gt_x_up_O_s3[:, 0].to(torch.int16).cpu()
            
            # Encode all stages
            byte_stream_s0 = torchac.encode_int16_normalized_cdf(x_up_O_cdf_norm_s0, gt_x_up_O_s0_cpu)
            byte_stream_s1 = torchac.encode_int16_normalized_cdf(x_up_O_cdf_norm_s1, gt_x_up_O_s1_cpu)
            byte_stream_s2 = torchac.encode_int16_normalized_cdf(x_up_O_cdf_norm_s2, gt_x_up_O_s2_cpu)
            byte_stream_s3 = torchac.encode_int16_normalized_cdf(x_up_O_cdf_norm_s3, gt_x_up_O_s3_cpu)
            
            # Add to byte stream list
            byte_stream_ls.append(byte_stream_s0)
            byte_stream_ls.append(byte_stream_s1)
            byte_stream_ls.append(byte_stream_s2)
            byte_stream_ls.append(byte_stream_s3)
        
        # Pack all byte streams into one
        byte_stream = op.pack_byte_stream_ls(byte_stream_ls)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    enc_time_end = time.time()
    
    # Save base point cloud information
    base_x_coords, base_x_feats = data_ls[0]
    base_x_len = base_x_coords.shape[0] 
    base_x_coords = base_x_coords[:, 1:].cpu().numpy() # (n, 3)
    base_x_feats = base_x_feats.cpu().numpy() # (n, 1)
    
    # Write compressed file
    with open(output_path, 'wb') as f:
        f.write(np.array(posQ, dtype=np.float16).tobytes())
        f.write(np.array(base_x_len, dtype=np.int32).tobytes())
        f.write(np.array(base_x_coords, dtype=np.int32).tobytes())
        f.write(np.array(base_x_feats, dtype=np.uint8).tobytes())
        f.write(byte_stream)
    
    # Calculate bitrate and encoding time
    enc_time = enc_time_end - enc_time_start
    file_size_bits = op.get_file_size_in_bits(output_path)
    bpp = file_size_bits / N
    
    # Return compression result information
    return {
        'bpp': bpp,
        'enc_time': enc_time,
        'file_size_bits': file_size_bits,
        'num_points': N,
        'output_path': output_path
    }

# Usage example:
# xyz_quantized = input quantized point cloud coordinates, numpy array or torch tensor, shape (N, 3)
# result = compress_point_cloud(
#     xyz_quantized=xyz_quantized,
#     ckpt_path='./model/KITTIDetection/ckpt_ue_4stage_conv.pt',
#     output_path='./output/compressed.bin'
# )
# print(f"Compression complete | Bitrate: {result['bpp']:.3f} bpp | Encoding time: {result['enc_time']:.3f}s")



def decompress_point_cloud(
    bin_file_path,           # Path to compressed bin file
    ckpt_path,               # Path to pre-trained weights file
    output_path=None,        # Path for output ply file (optional)
    channels=32,             # Network channel count
    kernel_size=5,           # Convolution kernel size
    is_data_pre_quantized=True  # Whether original point cloud is pre-quantized
):
    """
    Decompress point cloud from bin file
    
    Parameters:
        bin_file_path: Path to compressed bin file
        ckpt_path: Path to pre-trained model weights file
        output_path: Path for output decompressed ply file. If None, only return point cloud data without saving
        channels: Neural network channel count
        kernel_size: Convolution kernel size
        is_data_pre_quantized: Whether original point cloud is pre-quantized
    
    Returns:
        dict: Dictionary containing decompression result info (decoding time, point count, decompressed point cloud data, etc.)
    """
    # If output path is specified, ensure output directory exists
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set torchsparse configuration
    conv_config = F.conv_config.get_default_conv_config()
    conv_config.kmap_mode = "hashmap"
    F.conv_config.set_global_conv_config(conv_config)
    
    # Import network model (needs to be imported in the file using this function)
    from network_ue_4stage_conv import Network
    
    # Load network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Network(channels=channels, kernel_size=kernel_size)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.to(device).eval()
    
    # Read bin file
    with open(bin_file_path, 'rb') as f:
        posQ = np.frombuffer(f.read(2), dtype=np.float16)[0]
        base_x_len = np.frombuffer(f.read(4), dtype=np.int32)[0]
        base_x_coords = np.frombuffer(f.read(base_x_len*4*3), dtype=np.int32)
        base_x_feats = np.frombuffer(f.read(base_x_len*1), dtype=np.uint8)
        byte_stream = f.read()
    
    # Start decoding
    torch.cuda.synchronize() if device == 'cuda' else None
    dec_time_start = time.time()
    
    with torch.no_grad():
        # Prepare base point cloud
        base_x_coords = torch.tensor(base_x_coords.reshape(-1, 3), device=device) 
        base_x_feats = torch.tensor(base_x_feats.reshape(-1, 1), device=device)
        
        # Initialize x as base point cloud
        x = SparseTensor(coords=torch.cat((base_x_feats*0, base_x_coords), dim=-1), feats=base_x_feats).to(device)
        
        # Unpack byte stream
        byte_stream_ls = op.unpack_byte_stream(byte_stream)
        
        for byte_stream_idx in range(0, len(byte_stream_ls), 4):  # Each loop processes 4 stages of byte streams
            byte_stream_s0 = byte_stream_ls[byte_stream_idx]     # Byte stream for 1st bit
            byte_stream_s1 = byte_stream_ls[byte_stream_idx+1]   # Byte stream for 2nd bit
            byte_stream_s2 = byte_stream_ls[byte_stream_idx+2]   # Byte stream for 3rd-4th bits
            byte_stream_s3 = byte_stream_ls[byte_stream_idx+3]   # Byte stream for 5th-8th bits
            
            # Embedding prior scale features
            x_O = x.feats.int()
            x_F = net.prior_embedding(x_O).view(-1, net.channels) # (N_d, C)
            x = SparseTensor(coords=x.coords, feats=x_F)
            x = net.prior_resnet(x) # (N_d, C)
            
            # Target embedding
            x_up_C, x_up_F = net.fcg(x.coords, x_O, x_F=x.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)
            
            x_up_F = net.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = net.target_resnet(x_up)
            
            # First stage decoding - 1st bit
            x_up_s0 = net.spatial_conv_s0(x_up)
            x_up_O_prob_s0 = net.pred_head_s0(x_up_s0.feats)  # (N, 2)
            x_up_O_cdf_s0 = torch.cat((x_up_O_prob_s0[:, 0:1]*0, x_up_O_prob_s0.cumsum(dim=-1)), dim=-1)
            x_up_O_cdf_s0 = torch.clamp(x_up_O_cdf_s0, min=0, max=1)
            x_up_O_cdf_s0_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s0, True)
            x_up_O_cdf_s0_norm = x_up_O_cdf_s0_norm.cpu()
            
            # Decode 1st bit
            x_up_O_s0 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s0_norm, byte_stream_s0).to(device)
            
            # Second stage decoding - 2nd bit
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
            x_up_O_s1 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s1_norm, byte_stream_s1).to(device)
            
            # Third stage decoding - 3rd-4th bits
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
            x_up_O_s2 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s2_norm, byte_stream_s2).to(device)
            
            # Fourth stage decoding - 5th-8th bits
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
            x_up_O_s3 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s3_norm, byte_stream_s3).to(device)
            
            # Combine all bits to restore complete 8-bit value
            x_up_O = (x_up_O_s0 * 128 + x_up_O_s1 * 64 + x_up_O_s2 * 16 + x_up_O_s3).unsqueeze(-1)
            
            # Update current layer's sparse tensor
            x = SparseTensor(coords=x_up_C, feats=x_up_O).to(device)
        
        # Call fcg to get the final point cloud
        scan = net.fcg(x.coords, x.feats.int())
        
        # Process point cloud coordinates
        if is_data_pre_quantized:
            scan = scan[:, 1:] * posQ
        else:
            scan = (scan[:, 1:] * posQ - 131072) * 0.001
    
    torch.cuda.synchronize() if device == 'cuda' else None
    dec_time_end = time.time()
    dec_time = dec_time_end - dec_time_start
    
    # Convert result to numpy array
    point_cloud = scan
    
    # Save as PLY file if needed
    if output_path:
        io.save_ply_ascii_geo(point_cloud, output_path)
    
    # Return decompression results
    return {
        'dec_time': dec_time,
        'num_points': point_cloud.shape[0],
        'point_cloud': point_cloud,
        'output_path': output_path
    }

# Usage example:
# result = decompress_point_cloud(
#     bin_file_path='./data/compressed.bin',
#     ckpt_path='./model/KITTIDetection/ckpt_ue_4stage_conv.pt',
#     output_path='./data/decompressed.ply'
# )
# print(f"Decompression complete | Points: {result['num_points']} | Decoding time: {result['dec_time']:.3f}s")