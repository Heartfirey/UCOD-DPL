"""
Discriminator module for adversarial training.

This module implements a discriminator network that can be used for
adversarial training in image segmentation tasks.
"""

import torch
from engine.config.config import CfgNode
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """
    A convolutional block with batch normalization and leaky ReLU activation.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        padding (int): Zero-padding added to both sides of the input
        leaky_relu_slope (float): Negative slope of the LeakyReLU activation
        bias (bool): If True, adds a learnable bias to the output
        zero_init (bool): If True, initializes convolution weights to zero
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        padding: int, 
        leaky_relu_slope: float = 0.1, 
        bias: bool = False, 
        zero_init: bool = False
    ):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True)
        )
        
        if zero_init:
            self._zero_initialize_conv_layers()
    
    def _zero_initialize_conv_layers(self) -> None:
        """Initialize convolution layers with zero weights and biases."""
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.weight, 0)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor after convolution, batch norm, and activation
        """
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, config: CfgNode):
        super().__init__()
        self.maskConv = ConvBlock(1, 32, 3, 1, 1)
        self.use_features = config.dis_use_features
        if self.use_features:
            self.featureConv = ConvBlock(config.dim, config.dim, 3, 1, 1)
        indim = self.use_features * config.dim + 32
        outdim = indim // 2
        self.convs = nn.ModuleList([ConvBlock(indim // (2**i), outdim // (2**i), kernel_size=3, stride=2, padding=1) for i in range(2)])
        self.linear = nn.Linear(outdim// 2 * ((config.feature_size + 3)//4) ** 2 , 1)
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, mask, feature):
        input = self.maskConv(mask)
        if self.use_features:  
            feature = self.featureConv(feature)
            input = torch.cat((input,feature), dim=1)
        for blk in self.convs:
            input = blk(input)
        input = torch.flatten(input, 1)
        x = self.linear(input)
        return torch.sigmoid(x)