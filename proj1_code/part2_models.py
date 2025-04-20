#!/usr/bin/python3

"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proj1_code.part1 import create_Gaussian_kernel_2D


class HybridImageModel(nn.Module):
    def __init__(self):
        """
        Initializes an instance of the HybridImageModel class.
        """
        super().__init__()

    def get_kernel(self, cutoff_frequency: int) -> torch.Tensor:
        """
        Returns a Gaussian kernel using the specified cutoff frequency.

        PyTorch requires the kernel to be of a particular shape in order to
        apply it to an image. Specifically, the kernel needs to be of shape
        (c, 1, k, k) where c is the # channels in the image. Start by getting a
        2D Gaussian kernel using your implementation from Part 1, which will be
        of shape (k, k). Then, let's say you have an RGB image, you will need to
        turn this into a Tensor of shape (3, 1, k, k) by stacking the Gaussian
        kernel 3 times.

        Args
            cutoff_frequency: int specifying cutoff_frequency
        Returns
            kernel: Tensor of shape (c, 1, k, k) where c is # channels

        HINTS:
        - You will use the create_Gaussian_kernel_2D() function from part1.py in
          this function.
        - Since the # channels may differ across each image in the dataset,
          make sure you don't hardcode the dimensions you reshape the kernel
          to. There is a variable defined in this class to give you channel
          information.
        - You can use np.reshape() to change the dimensions of a numpy array.
        - You can use np.tile() to repeat a numpy array along specified axes.
        - You can use torch.Tensor() to convert numpy arrays to torch Tensors.
        """

        ############################
        ### TODO: YOUR CODE HERE ###

        # 获取2D高斯核
        gaussian_kernel_2d = create_Gaussian_kernel_2D(cutoff_frequency)
        
        # 添加维度，使形状为(1, 1, k, k)
        kernel_reshaped = np.reshape(gaussian_kernel_2d, (1, 1, gaussian_kernel_2d.shape[0], gaussian_kernel_2d.shape[1]))
        
        # 根据通道数重复核，得到(c, 1, k, k)的形状
        kernel_tiled = np.tile(kernel_reshaped, (self.n_channels, 1, 1, 1))
        
        # 转换为PyTorch Tensor
        kernel = torch.Tensor(kernel_tiled)

        ### END OF STUDENT CODE ####
        ############################

        return kernel

    def low_pass(self, x: torch.Tensor, kernel: torch.Tensor):
        """
        Applies low pass filter to the input image.

        Args:
            x: Tensor of shape (b, c, m, n) where b is batch size
            kernel: low pass filter to be applied to the image
        Returns:
            filtered_image: Tensor of shape (b, c, m, n)

        HINTS:
        - You should use the 2d convolution operator from torch.nn.functional.
        - Make sure to pad the image appropriately (it's a parameter to the
          convolution function you should use here!).
        - Pass self.n_channels as the value to the "groups" parameter of the
          convolution function. This represents the # of channels that the
          filter will be applied to.
        """

        ############################
        ### TODO: YOUR CODE HERE ###

        # 获取卷积核的大小
        kernel_size = kernel.shape[2]
        
        # 计算需要的padding大小
        # 为了保持输出图像和输入图像大小一致，padding = kernel_size // 2
        padding = kernel_size // 2
        
        # 使用F.conv2d进行低通滤波，groups=self.n_channels使每个通道单独卷积
        filtered_image = F.conv2d(
            x,                   # 输入图像
            kernel,              # 卷积核
            padding=padding,     # 适当的padding
            groups=self.n_channels  # 每个通道独立应用滤波器
        )

        ### END OF STUDENT CODE ####
        ############################

        return filtered_image

    def forward(
        self, image1: torch.Tensor, image2: torch.Tensor, cutoff_frequency: torch.Tensor
    ):
        """
        Takes two images and creates a hybrid image. Returns the low frequency
        content of image1, the high frequency content of image 2, and the
        hybrid image.

        Args:
            image1: Tensor of shape (b, c, m, n)
            image2: Tensor of shape (b, c, m, n)
            cutoff_frequency: Tensor of shape (b)
        Returns:
            low_frequencies: Tensor of shape (b, c, m, n)
            high_frequencies: Tensor of shape (b, c, m, n)
            hybrid_image: Tensor of shape (b, c, m, n)

        HINTS:
        - You will use the get_kernel() function and your low_pass() function
          in this function.
        - Similar to Part 1, you can get just the high frequency content of an
          image by removing its low frequency content.
        - Don't forget to make sure to clip the pixel values >=0 and <=1. You
          can use torch.clamp().
        - If you want to use images with different dimensions, you should
          resize them in the HybridImageDataset class using
          torchvision.transforms.
        """
        self.n_channels = image1.shape[1]

        ############################
        ### TODO: YOUR CODE HERE ###

        # 处理批次中的每个图像
        batch_size = image1.shape[0]
        
        # 初始化结果张量
        low_frequencies = torch.zeros_like(image1)
        high_frequencies = torch.zeros_like(image2)
        hybrid_image = torch.zeros_like(image1)
        
        # 对批次中的每个图像单独处理（因为每个图像可能有不同的截止频率）
        for i in range(batch_size):
            # 获取当前图像的截止频率
            curr_cutoff_freq = int(cutoff_frequency[i].item())
            
            # 获取对应的高斯核
            kernel = self.get_kernel(curr_cutoff_freq)
            
            # 提取当前批次的图像
            curr_image1 = image1[i:i+1]
            curr_image2 = image2[i:i+1]
            
            # 获取image1的低频内容
            curr_low_frequencies = self.low_pass(curr_image1, kernel)
            
            # 获取image2的高频内容（原始image2减去低频内容）
            curr_image2_low_freq = self.low_pass(curr_image2, kernel)
            curr_high_frequencies = curr_image2 - curr_image2_low_freq
            
            # 创建混合图像
            curr_hybrid_image = curr_low_frequencies + curr_high_frequencies
            
            # 将结果存储在输出张量中
            low_frequencies[i:i+1] = curr_low_frequencies
            high_frequencies[i:i+1] = curr_high_frequencies
            hybrid_image[i:i+1] = curr_hybrid_image
        
        hybrid_image = torch.clamp(hybrid_image, 0, 1)

        ### END OF STUDENT CODE ####
        ############################

        return low_frequencies, high_frequencies, hybrid_image
