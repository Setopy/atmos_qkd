"""
models/tcn.py
-------------
Temporal Convolutional Network for Zernike sequence prediction.

A TCN uses dilated causal convolutions to model temporal dependencies.
Dilation means the receptive field grows exponentially with network
depth — a 4-layer network with dilation factors [1, 2, 4, 8] covers
long history with few parameters.

Causal convolutions ensure prediction at time t uses only past inputs.

Architecture
------------
Input : (batch, seq_len, in_features)
Output: (batch, out_features)

Reference: Bai et al. (2018). arXiv:1803.01271.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv    = nn.Conv1d(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 dilation=dilation, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1   = CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
        self.conv2   = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1   = nn.LayerNorm(out_ch)
        self.norm2   = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()
        self.res_conv = (nn.Conv1d(in_ch, out_ch, 1)
                         if in_ch != out_ch else None)

    def forward(self, x):
        res = x
        out = self.relu(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(out).transpose(1,2)).transpose(1,2))
        out = self.dropout(out)
        if self.res_conv is not None:
            res = self.res_conv(res)
        return self.relu(out + res)


class ZernikeTCN(nn.Module):
    """
    TCN that maps a window of past Zernike+context vectors to the
    next Zernike coefficient vector.

    Parameters
    ----------
    in_features  : int   features per time step in input sequence
    out_features : int   number of output values
    window_size  : int   input sequence length
    channels     : int   internal channel width
    n_blocks     : int   number of residual blocks
    kernel_size  : int   conv kernel width
    dropout      : float dropout probability
    """

    def __init__(self, in_features=18, out_features=15,
                 window_size=60, channels=64, n_blocks=4,
                 kernel_size=3, dropout=0.1):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.window_size  = window_size

        self.input_proj = nn.Linear(in_features, channels)
        self.tcn_blocks = nn.Sequential(*[
            TCNResidualBlock(channels, channels,
                             kernel_size, 2**i, dropout)
            for i in range(n_blocks)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, out_features)
        )

    def forward(self, x):
        out = self.input_proj(x).transpose(1, 2)
        out = self.tcn_blocks(out)
        return self.output_head(out[:, :, -1])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
