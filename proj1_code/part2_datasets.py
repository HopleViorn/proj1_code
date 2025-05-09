#!/usr/bin/python3

"""
PyTorch tutorial on data loading & processing:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import os
from typing import List, Tuple

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms


def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Creates a dataset of paired images from a directory.

    The dataset should be partitioned into two sets: one contains images that
    will have the low pass filter applied, and the other contains images that
    will have the high pass filter applied.

    Args:
        path: string specifying the directory containing images
    Returns:
        images_a: list of strings specifying the paths to the images in set A,
           in lexicographically-sorted order
        images_b: list of strings specifying the paths to the images in set B,
           in lexicographically-sorted order
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    
    images_a = []
    images_b = []
    
    files = sorted(os.listdir(path))
    
    for file in files:
        if 'a_' in file:
            images_a.append(os.path.join(path, file))
        elif 'b_' in file:
            images_b.append(os.path.join(path, file))
    
    images_a.sort()
    images_b.sort()

    ### END OF STUDENT CODE ####
    ############################

    return images_a, images_b


def get_cutoff_frequencies(path: str) -> List[int]:
    """
    Gets the cutoff frequencies corresponding to each pair of images.

    The cutoff frequencies are the values you discovered from experimenting in
    part 1.

    Args:
        path: string specifying the path to the .txt file with cutoff frequency
        values
    Returns:
        cutoff_frequencies: numpy array of ints. The array should have the same
            length as the number of image pairs in the dataset
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    
    cutoff_frequencies = []
    
    # 读取文本文件
    with open(path, 'r') as file:
        for line in file:
            if line.strip():
                cutoff_frequencies.append(int(line.strip()))
    
    # 将列表转换为numpy数组
    cutoff_frequencies = np.array(cutoff_frequencies)

    ### END OF STUDENT CODE ####
    ############################

    return cutoff_frequencies


class HybridImageDataset(data.Dataset):
    """Hybrid images dataset."""

    def __init__(self, image_dir: str, cf_file: str) -> None:
        """
        HybridImageDataset class constructor.

        You must replace self.transform with the appropriate transform from
        torchvision.transforms that converts a PIL image to a torch Tensor. You
        can specify additional transforms (e.g. image resizing) if you want to,
        but it's not necessary for the images we provide you since each pair has
        the same dimensions.

        Args:
            image_dir: string specifying the directory containing images
            cf_file: string specifying the path to the .txt file with cutoff
            frequency values
        """
        images_a, images_b = make_dataset(image_dir)
        cutoff_frequencies = get_cutoff_frequencies(cf_file)

        self.transform = None
        ############################
        ### TODO: YOUR CODE HERE ###
        
        self.transform = transforms.ToTensor()

        ### END OF STUDENT CODE ####
        ############################

        self.images_a = images_a
        self.images_b = images_b
        self.cutoff_frequencies = cutoff_frequencies

    def __len__(self) -> int:
        """Returns number of pairs of images in dataset."""

        ############################
        ### TODO: YOUR CODE HERE ###
        
        return len(self.images_a)

        ### END OF STUDENT CODE ####
        ############################

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns the pair of images and corresponding cutoff frequency value at
        index `idx`.

        Since self.images_a and self.images_b contain paths to the images, you
        should read the images here and normalize the pixels to be between 0
        and 1. Make sure you transpose the dimensions so that image_a and
        image_b are of shape (c, m, n) instead of the typical (m, n, c), and
        convert them to torch Tensors.

        Args:
            idx: int specifying the index at which data should be retrieved
        Returns:
            image_a: Tensor of shape (c, m, n)
            image_b: Tensor of shape (c, m, n)
            cutoff_frequency: int specifying the cutoff frequency corresponding
               to (image_a, image_b) pair

        HINTS:
        - You should use the PIL library to read images
        - You will use self.transform to convert the PIL image to a torch Tensor
        """

        ############################
        ### TODO: YOUR CODE HERE ###
        
        # 获取指定索引的图像路径
        image_path_a = self.images_a[idx]
        image_path_b = self.images_b[idx]
        
        # 使用PIL库读取图像
        pil_image_a = PIL.Image.open(image_path_a)
        pil_image_b = PIL.Image.open(image_path_b)
        
        # 使用transform将PIL图像转换为torch张量
        # transform会自动将像素值归一化到[0,1]范围，并转置维度为(c,m,n)
        image_a = self.transform(pil_image_a)
        image_b = self.transform(pil_image_b)
        
        # 获取对应的截止频率
        cutoff_frequency = self.cutoff_frequencies[idx]

        ### END OF STUDENT CODE ####
        ############################

        return image_a, image_b, cutoff_frequency
