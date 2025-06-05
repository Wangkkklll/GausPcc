# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

import torch
import torch.nn as nn

class MahalanobisDistance(torch.nn.Module):
    def __init__(self, covariance_matrix):
        super(MahalanobisDistance, self).__init__()
        self.covariance_matrix = covariance_matrix

    def forward(self, input, target):
        # 计算马氏距离
        diff = input - target
        # 扩展协方差矩阵以便与批处理数据相匹配
        extended_cov_matrix = self.covariance_matrix.unsqueeze(0).expand(input.size(0), -1, -1).cuda()
        # 计算马氏距离
        temp = torch.diagonal(torch.matmul(torch.matmul(diff.unsqueeze(1), torch.inverse(extended_cov_matrix)), diff.unsqueeze(2)), dim1=1, dim2=2)
        # if torch.min(temp) > 0:
        #     print(temp)
        mahalanobis_dists = torch.mean(torch.sqrt(temp))
        # print(mahalanobis_dists)
        return mahalanobis_dists



class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, cov=None, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.lmbda = lmbda
        
        self.MahalanobisDistance = MahalanobisDistance(cov)
        

    def forward(self, output, attribute, mode='f'):
        out = {}
        if output.shape != attribute.shape:
            print(output.shape)
            print(attribute.shape)
        
        
        out["mse_loss"] = self.mse(output, attribute)
        
        if mode == 's':
            out["mse_loss"] = self.MahalanobisDistance(output, attribute)
        
        
        # out["maha_dist"] = self.MahalanobisDistance(output, attribute)        
        

        return out
