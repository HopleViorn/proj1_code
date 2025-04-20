#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def my_conv2d_pytorch(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Applies input filter(s) to the input image.

    Args:
        image: Tensor of shape (1, d1, h1, w1)
        kernel: Tensor of shape (N, d1/groups, k, k) to be applied to the image
    Returns:
        filtered_image: Tensor of shape (1, d2, h2, w2) where
           d2 = N
           h2 = (h1 - k + 2 * padding) / stride + 1
           w2 = (w1 - k + 2 * padding) / stride + 1

    HINTS:
    - You should use the 2d convolution operator from torch.nn.functional.
    - In PyTorch, d1 is `in_channels`, and d2 is `out_channels`
    - Make sure to pad the image appropriately (it's a parameter to the
      convolution function you should use here!).
    - You can assume the number of groups is equal to the number of input channels.
    - You can assume only square filters for this function.
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    # Get input dimensions
    _, _, h1, w1 = image.shape
    N, _, k, _ = kernel.shape
    
    # Calculate output dimensions
    padding = k // 2  # Same padding
    stride = 1  # Default stride
    
    # Apply convolution
    filtered_image = F.conv2d(
        input=image,
        weight=kernel,
        bias=None,
        stride=stride,
        padding=padding,
        groups=image.size(1)  # groups equals input channels
    )
    
    ### END OF STUDENT CODE ####
    ############################

    return filtered_image
