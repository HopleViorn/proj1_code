#!/usr/bin/python3

from typing import Tuple
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from utils import load_image, save_image

def frequency_compression(img: np.ndarray, ratio: float) -> Tuple[np.ndarray, float]:
    """
    Perform frequency-based compression on image
    Args:
        img: input image (H,W,C) in [0,1] range
        ratio: retention ratio of low frequencies (0-1)
    Returns:
        compressed_img: reconstructed image
        psnr: PSNR between original and compressed image
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    
    # Compute 2D FFT and shift zero frequency to center
    f = fft.fft2(img)
    fshift = fft.fftshift(f)
    
    # Create low-pass mask
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    # Calculate radius based on retention ratio
    radius = int(min(rows, cols) * ratio / 2)
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1
    
    # Apply mask and inverse FFT
    fshift_filtered = fshift * mask
    f_ishift = fft.ifftshift(fshift_filtered)
    img_back = fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Calculate PSNR
    psnr = calculate_psnr(img, img_back)
    
    return img_back, psnr

def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate PSNR between original and compressed image
    Args:
        original: original image
        compressed: compressed/reconstructed image
    Returns:
        psnr: PSNR value in dB
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # since images are in [0,1] range
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def visualize_results(img_path: str, ratios: list = [0.1, 0.3, 0.5, 0.7]):
    """
    Visualize compression results for different retention ratios
    Args:
        img_path: path to input image
        ratios: list of retention ratios to test
    """
    # Load image
    img = load_image(img_path)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)  # convert to grayscale
    
    plt.figure(figsize=(12, 8))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Process each ratio
    for i, ratio in enumerate(ratios):
        compressed_img, psnr = frequency_compression(img, ratio)
        
        plt.subplot(2, 3, i+2)
        plt.imshow(compressed_img, cmap='gray')
        plt.title(f'Ratio={ratio}\nPSNR={psnr:.2f}dB')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../results/part4/compression_results.png')
    plt.show()

def main():
    # Example usage
    img_path = "../data/1a_dog.bmp"
    visualize_results(img_path)

if __name__ == "__main__":
    main()