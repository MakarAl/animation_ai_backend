"""
Image I/O utilities for Animation AI Backend.
Provides functions for loading, resizing, and saving images with consistent naming.
"""

import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Tuple, Optional, Union
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """
    Load an image from path and convert to RGB numpy array.
    
    Args:
        path: Path to the image file
        
    Returns:
        RGB image as numpy array with shape (H, W, 3)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Load with PIL for better format support
    img = Image.open(path).convert('RGB')
    return np.array(img)


def resize_image(img: np.ndarray, size: Union[int, Tuple[int, int]], 
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to specified size.
    
    Args:
        img: Input image as numpy array
        size: Target size (width) or (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio when only width is specified
        
    Returns:
        Resized image as numpy array
    """
    if isinstance(size, int):
        # Only width specified, calculate height to maintain aspect ratio
        if keep_aspect_ratio:
            h, w = img.shape[:2]
            new_w = size
            new_h = int(h * new_w / w)
            size = (new_w, new_h)
        else:
            size = (size, size)
    
    # Convert to PIL for resizing
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(size, Image.LANCZOS)
    return np.array(pil_img)


def save_image(img: np.ndarray, output_folder: str, model_name: str, 
               fmt: str = 'png', prefix: str = '') -> str:
    """
    Save image with auto-generated filename.
    
    Args:
        img: Image to save as numpy array
        output_folder: Directory to save the image
        model_name: Name of the model that generated the image
        fmt: Output format ('png', 'jpg', 'gif')
        prefix: Optional prefix for filename
        
    Returns:
        Path to the saved image
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    if prefix:
        filename = f"{prefix}_{timestamp}_{model_name}.{fmt}"
    else:
        filename = f"{timestamp}_{model_name}.{fmt}"
    
    output_path = os.path.join(output_folder, filename)
    
    # Save image
    if fmt.lower() == 'jpg' or fmt.lower() == 'jpeg':
        # Convert to BGR for OpenCV
        if img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        cv2.imwrite(output_path, img_bgr)
    else:
        # Use PIL for other formats
        pil_img = Image.fromarray(img)
        pil_img.save(output_path)
    
    return output_path


def save_gif(images: list, output_path: str, duration: float = 0.5) -> str:
    """
    Save a list of images as a GIF.
    
    Args:
        images: List of images as numpy arrays
        output_path: Path to save the GIF
        duration: Duration per frame in seconds
        
    Returns:
        Path to the saved GIF
    """
    # Convert numpy arrays to PIL Images
    pil_images = [Image.fromarray(img) for img in images]
    
    # Save as GIF
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=int(duration * 1000),  # Convert to milliseconds
        loop=0
    )
    
    return output_path


def get_image_info(path: str) -> dict:
    """
    Get basic information about an image.
    
    Args:
        path: Path to the image file
        
    Returns:
        Dictionary with image information
    """
    img = Image.open(path)
    return {
        'width': img.width,
        'height': img.height,
        'mode': img.mode,
        'format': img.format,
        'size_bytes': os.path.getsize(path)
    } 