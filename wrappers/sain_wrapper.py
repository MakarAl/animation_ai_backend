import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import argparse
from typing import Optional, Tuple
import imageio
from PIL import ImageOps
from datetime import datetime
import cv2

# Set up model directory for consolidated backend
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(current_dir), 'models', 'SAIN')
sys.path.insert(0, models_dir)

from SAIN import SAIN

sain_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'SAIN')
if sain_dir not in sys.path:
    sys.path.insert(0, sain_dir)


def get_best_device():
    """
    Automatically detect the best available device.
    Returns 'cuda' if available, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_resize_and_pad_dims(orig_w, orig_h, max_side, multiple=64):
    """
    Compute new size and padding to fit within max_side, preserve aspect ratio, and make divisible by `multiple`.
    Returns (new_w, new_h, pad_left, pad_top, pad_right, pad_bottom, new_w_aligned, new_h_aligned)
    """
    scale = min(max_side / orig_w, max_side / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    # Make divisible by `multiple`
    new_w_aligned = ((new_w + multiple - 1) // multiple) * multiple
    new_h_aligned = ((new_h + multiple - 1) // multiple) * multiple
    pad_left = (new_w_aligned - new_w) // 2
    pad_right = new_w_aligned - new_w - pad_left
    pad_top = (new_h_aligned - new_h) // 2
    pad_bottom = new_h_aligned - new_h - pad_top
    return new_w, new_h, pad_left, pad_top, pad_right, pad_bottom, new_w_aligned, new_h_aligned


def pad_img(img, pad_left, pad_top, pad_right, pad_bottom, fill=0):
    return ImageOps.expand(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)


class SAINWrapper:
    """
    Wrapper class for SAIN (Sketch-Aware Interpolation Network) model.
    Normalized for the consolidated Animation AI backend.
    """
    
    def __init__(self, 
                 checkpoint_path: str = None,
                 device: Optional[str] = None,
                 size: int = 512):
        """
        Initialize the SAIN wrapper.
        
        Args:
            checkpoint_path: Path to the model checkpoint (defaults to consolidated backend path)
            device: Device to run the model on ('cuda', 'mps', 'cpu', or None for auto-detect)
            size: Input image size (max side, must be divisible by 8)
        """
        # Set default checkpoint path for consolidated backend
        if checkpoint_path is None:
            checkpoint_path = os.path.join(models_dir, 'ckp', 'checkpoints', 'model_best.pth')
        
        # Auto-detect device if not specified
        if device is None:
            device = get_best_device()
            print(f"Auto-detected device: {device}")
        
        # Validate device
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = "cpu"
        
        self.device = torch.device(device)
        self.size = size
        
        # Validate size is divisible by 8
        if size % 8 != 0:
            raise ValueError(f"Size must be divisible by 8, got {size}")
        
        # Load RAFT optical flow model
        print(f"Loading RAFT optical flow model on {device}...")
        try:
            self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device).eval()
        except Exception as e:
            print(f"Error loading RAFT model: {e}")
            print("Trying to load on CPU...")
            self.device = torch.device("cpu")
            self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device).eval()
        
        # Initialize SAIN model
        print("Loading SAIN model...")
        self._init_sain_model(checkpoint_path)
        
        print(f"SAIN wrapper initialized successfully on {self.device}!")
    
    def _init_sain_model(self, checkpoint_path: str):
        """Initialize the SAIN model and load checkpoint."""
        # Create model arguments
        sargs = type("", (), {})()
        sargs.device = self.device
        sargs.phase = "test"
        sargs.crop_size = self.size
        sargs.joinType = "concat"
        sargs.c = 64
        sargs.window_size = 24
        sargs.resume_flownet = ""
        
        # Create model
        self.model = SAIN(sargs).to(self.device).eval()
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        
        # Clean state dict
        newst = {}
        for k, v in state.items():
            k2 = k
            for pref in ("module.", "sain."):
                if k2.startswith(pref):
                    k2 = k2[len(pref):]
            if "attn_mask" in k2:
                continue
            newst[k2] = v
        
        self.model.load_state_dict(newst, strict=False)
        print("Checkpoint loaded successfully!")
    
    def _preprocess_image(self, image_path: str, target_w: int, target_h: int, pad: Tuple[int,int,int,int]) -> torch.Tensor:
        """
        Preprocess an image for the model.
        
        Args:
            image_path: Path to the input image
            target_w: Target width
            target_h: Target height
            pad: Padding tuple (left, top, right, bottom)
            
        Returns:
            Preprocessed tensor
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize((target_w, target_h), resample=Image.BICUBIC)
        img = pad_img(img, *pad)
        tensor = to_tensor(img).unsqueeze(0).sub_(0.5).div_(0.5)
        return tensor.to(self.device)
    
    def _postprocess_tensor(self, tensor: torch.Tensor, crop_box: Tuple[int,int,int,int]) -> np.ndarray:
        """
        Postprocess model output tensor to numpy array.
        
        Args:
            tensor: Model output tensor
            crop_box: Crop box tuple (left, top, right, bottom)
            
        Returns:
            Postprocessed numpy array
        """
        if tensor.ndim == 4:
            tensor = tensor[0]
        
        # Denormalize from [-1, 1] to [0, 1]
        tensor = tensor.add(1).div(2).clamp(0, 1)
        
        # Convert to numpy array
        arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        
        # Crop to original region
        left, top, right, bottom = crop_box
        arr = arr[top:bottom, left:right, :]
        
        return arr
    
    def interpolate(self, 
                   img0_path: str, 
                   img1_path: str, 
                   t: float = 0.5) -> Optional[np.ndarray]:
        """
        Generate an inbetween frame from two input images.
        
        Args:
            img0_path: Path to the first input image
            img1_path: Path to the second input image
            t: Interpolation time (0.0 to 1.0, default 0.5 for middle frame)
            
        Returns:
            Inbetween frame as numpy array, or None if failed
        """
        try:
            # Load and get original dimensions
            img0_orig = Image.open(img0_path).convert("RGB")
            img1_orig = Image.open(img1_path).convert("RGB")
            orig_w, orig_h = img0_orig.size
            
            # Calculate resize and padding dimensions
            new_w, new_h, pad_left, pad_top, pad_right, pad_bottom, target_w, target_h = get_resize_and_pad_dims(
                orig_w, orig_h, self.size
            )
            
            # Preprocess images
            img0_tensor = self._preprocess_image(img0_path, new_w, new_h, (pad_left, pad_top, pad_right, pad_bottom))
            img1_tensor = self._preprocess_image(img1_path, new_w, new_h, (pad_left, pad_top, pad_right, pad_bottom))
            
            # Create points tensor (attention mask)
            points = torch.ones((1, 1, target_h, target_w), device=self.device)
            
            # Compute optical flow using RAFT
            print("Computing optical flow...")
            inp0 = to_tensor(pad_img(img0_orig.resize((new_w, new_h), resample=Image.BICUBIC), pad_left, pad_top, pad_right, pad_bottom)).unsqueeze(0).to(self.device)
            inp1 = to_tensor(pad_img(img1_orig.resize((new_w, new_h), resample=Image.BICUBIC), pad_left, pad_top, pad_right, pad_bottom)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                flowAB = self.raft(inp0, inp1)[-1]
                flowBA = self.raft(inp1, inp0)[-1]
                region_flow = [flowAB, flowBA]
            
            # Run SAIN model
            print("Running SAIN model...")
            with torch.no_grad():
                output = self.model(img0_tensor, img1_tensor, points, region_flow)
            
            # Postprocess output
            crop_box = (pad_left, pad_top, pad_left + new_w, pad_top + new_h)
            result = self._postprocess_tensor(output, crop_box)
            
            return result
            
        except Exception as e:
            print(f"Error during interpolation: {e}")
            import traceback
            traceback.print_exc()
            return None


def sain_interpolate(img0_path: str,
                     img1_path: str,
                     t: float = 0.5,
                     checkpoint_path: str = None,
                     device: Optional[str] = None,
                     size: int = 512) -> Optional[np.ndarray]:
    """
    Convenience function for one-shot SAIN interpolation.
    
    Args:
        img0_path: Path to the first input image
        img1_path: Path to the second input image
        t: Interpolation time (0.0 to 1.0)
        checkpoint_path: Path to the model checkpoint
        device: Device to run the model on
        size: Input image size
        
    Returns:
        Inbetween frame as numpy array, or None if failed
    """
    wrapper = SAINWrapper(checkpoint_path=checkpoint_path, device=device, size=size)
    return wrapper.interpolate(img0_path, img1_path, t)


def main():
    """Example usage of the SAIN wrapper."""
    parser = argparse.ArgumentParser(description="SAIN interpolation example")
    parser.add_argument('img0', help='Path to first input image')
    parser.add_argument('img1', help='Path to second input image')
    parser.add_argument('--t', type=float, default=0.5, help='Interpolation timestamp (0.0 to 1.0)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None, help='Device to use')
    parser.add_argument('--size', type=int, default=512, help='Input image size (must be divisible by 8)')
    
    args = parser.parse_args()
    
    result = sain_interpolate(args.img0, args.img1, args.t, device=args.device, size=args.size)
    
    if result is not None:
        output_path = f"sain_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(output_path, result)
        print(f"Interpolated frame saved to: {output_path}")
    else:
        print("Interpolation failed!")


if __name__ == "__main__":
    main() 