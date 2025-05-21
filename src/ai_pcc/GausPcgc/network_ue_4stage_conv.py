import torch
import torch.nn as nn

import torchsparse
from torchsparse import nn as spnn
from torchsparse import SparseTensor

import kit.op as op
from kit.nn import ResNet, FOG, FCG, TargetEmbedding

class Network(nn.Module):
    def __init__(self, channels, kernel_size):
        super(Network, self).__init__()

        self.prior_embedding = nn.Embedding(256, channels)

        self.prior_resnet = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            ResNet(channels, k=kernel_size),
            ResNet(channels, k=kernel_size),
        )
        
        ###########################

        self.target_embedding = TargetEmbedding(channels)

        self.target_resnet = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            ResNet(channels, k=kernel_size),
            ResNet(channels, k=kernel_size),
        )

        ###########################

        # Add spatial convolution networks for each stage to capture neighborhood features
        # self.k = 5
        
        self.spatial_conv_s0 = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            spnn.Conv3d(channels, channels, kernel_size),
        )
        
        self.spatial_conv_s1 = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            spnn.Conv3d(channels, channels, kernel_size),
        )
        
        self.spatial_conv_s2 = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            spnn.Conv3d(channels, channels, kernel_size),
        )
        
        self.spatial_conv_s3 = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            spnn.Conv3d(channels, channels, kernel_size),
        )

        # Four-stage unequal prediction heads
        self.pred_head_s0 = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, 2),  # First bit, 2 possibilities
            nn.Softmax(dim=-1),
        )

        self.pred_head_s1_emb = nn.Embedding(2, channels)  # 1-bit embedding
        self.pred_head_s1 = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, 2),  # Second bit, 2 possibilities
            nn.Softmax(dim=-1),
        )

        self.pred_head_s2_emb = nn.Embedding(4, channels)  # 2-bit embedding (first two stages combined)
        self.pred_head_s2 = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, 4),  # Bits 3-4, 4 possibilities
            nn.Softmax(dim=-1),
        )

        self.pred_head_s3_emb = nn.Embedding(16, channels)  # 4-bit embedding (first three stages combined)
        self.pred_head_s3 = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, 16),  # Bits 5-8, 16 possibilities
            nn.Softmax(dim=-1),
        )

        self.channels = channels
        self.fog = FOG()
        self.fcg = FCG()

    def forward(self, x):
        N = x.coords.shape[0]

        # get sparse occupancy code list
        data_ls = []
        while True:
            x = self.fog(x)
            data_ls.append((x.coords.clone(), x.feats.clone())) # must clone
            if x.coords.shape[0] < 64:
                break
        data_ls = data_ls[::-1]
        # data_ls: [(coords, occupancy), (coords, occupancy), ...]

        total_bits = 0
        
        for depth in range(len(data_ls)-1):
            x_C, x_O = data_ls[depth]
            gt_x_up_C, gt_x_up_O = data_ls[depth+1]
            gt_x_up_C, gt_x_up_O = op.sort_CF(gt_x_up_C, gt_x_up_O)

            # embedding prior scale feats
            x_F = self.prior_embedding(x_O.int()).view(-1, self.channels) # (N_d, C)
            x = SparseTensor(coords=x_C, feats=x_F)
            x = self.prior_resnet(x) # (N_d, C) 

            # target embedding
            x_up_C, x_up_F = self.fcg(x_C, x_O, x.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)

            x_up_F = self.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = self.target_resnet(x_up)

            # Four-stage unequal bit encoding
            # stage1: 1st bit
            # stage2: 2nd bit
            # stage3: 3rd-4th bits
            # stage4: 5th-8th bits
            gt_x_up_O_s0 = torch.remainder(torch.div(gt_x_up_O, 128, rounding_mode='floor'), 2)  # 1st bit
            gt_x_up_O_s1 = torch.remainder(torch.div(gt_x_up_O, 64, rounding_mode='floor'), 2)   # 2nd bit
            gt_x_up_O_s2 = torch.remainder(torch.div(gt_x_up_O, 16, rounding_mode='floor'), 4)   # 3rd-4th bits
            gt_x_up_O_s3 = torch.remainder(gt_x_up_O, 16)                                        # 5th-8th bits

            # First stage prediction - 1st bit, with spatial convolution
            x_up_s0 = self.spatial_conv_s0(x_up)  # Utilize spatial neighborhood information
            x_up_O_prob_s0 = self.pred_head_s0(x_up_s0.feats)
            x_up_O_prob_s0 = x_up_O_prob_s0.gather(1, gt_x_up_O_s0.long())
            
            # Second stage prediction - 2nd bit (based on first stage result), with spatial convolution
            s0_emb = self.pred_head_s1_emb(gt_x_up_O_s0[:, 0].long())
            x_up_s1_F = x_up.feats + s0_emb
            x_up_s1 = SparseTensor(coords=x_up.coords, feats=x_up_s1_F)
            x_up_s1 = self.spatial_conv_s1(x_up_s1)  # Utilize spatial neighborhood information
            x_up_O_prob_s1 = self.pred_head_s1(x_up_s1.feats)
            x_up_O_prob_s1 = x_up_O_prob_s1.gather(1, gt_x_up_O_s1.long())
            
            # Third stage prediction - 3rd-4th bits (based on previous two stages), with spatial convolution
            prev_bits = gt_x_up_O_s0[:, 0] * 2 + gt_x_up_O_s1[:, 0]  # Combine first 2 bits
            s2_emb = self.pred_head_s2_emb(prev_bits.long())
            x_up_s2_F = x_up.feats + s2_emb
            x_up_s2 = SparseTensor(coords=x_up.coords, feats=x_up_s2_F)
            x_up_s2 = self.spatial_conv_s2(x_up_s2)  # Utilize spatial neighborhood information
            x_up_O_prob_s2 = self.pred_head_s2(x_up_s2.feats)
            x_up_O_prob_s2 = x_up_O_prob_s2.gather(1, gt_x_up_O_s2.long())
            
            # Fourth stage prediction - 5th-8th bits (based on previous three stages), with spatial convolution
            prev_bits = prev_bits * 4 + gt_x_up_O_s2[:, 0]  # Combine first 4 bits
            s3_emb = self.pred_head_s3_emb(prev_bits.long())
            x_up_s3_F = x_up.feats + s3_emb
            x_up_s3 = SparseTensor(coords=x_up.coords, feats=x_up_s3_F)
            x_up_s3 = self.spatial_conv_s3(x_up_s3)  # Utilize spatial neighborhood information
            x_up_O_prob_s3 = self.pred_head_s3(x_up_s3.feats)
            x_up_O_prob_s3 = x_up_O_prob_s3.gather(1, gt_x_up_O_s3.long())

            # Calculate total bits
            total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(x_up_O_prob_s0 + 1e-10), 0, 50))
            total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(x_up_O_prob_s1 + 1e-10), 0, 50))
            total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(x_up_O_prob_s2 + 1e-10), 0, 50))
            total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(x_up_O_prob_s3 + 1e-10), 0, 50))
            
        bpp = total_bits / N

        return bpp