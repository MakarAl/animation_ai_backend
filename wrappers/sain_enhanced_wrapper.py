import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from torchvision.transforms.functional import to_tensor, normalize
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import argparse
from typing import Optional, Tuple
import imageio
import cv2
from datetime import datetime

# Set up model directory for consolidated backend
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(current_dir), 'models', 'SAIN')
sys.path.insert(0, models_dir)

from SAIN import SAIN


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


def enhance_strokes_for_flow(img, dilation_radius=2, noise_amplitude=0.05):
    """
    Enhance line drawings for better optical flow computation.
    
    Args:
        img: PIL Image (line drawing)
        dilation_radius: Radius for stroke dilation
        noise_amplitude: Amplitude of synthetic noise to add
    
    Returns:
        Enhanced PIL Image
    """
    # Convert to numpy array
    img_array = np.array(img)
    
    # Convert to grayscale if RGB
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Create stroke mask (invert so strokes are white)
    stroke_mask = (gray < 128).astype(np.uint8) * 255
    
    # Dilate strokes to give them thickness
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius*2+1, dilation_radius*2+1))
    dilated = cv2.dilate(stroke_mask, kernel, iterations=1)
    
    # Apply bilateral filter to smooth edges while preserving structure
    filtered = cv2.bilateralFilter(dilated, 9, 75, 75)
    
    # Add subtle synthetic noise to provide texture for optical flow
    noise = np.random.normal(0, noise_amplitude * 255, gray.shape).astype(np.float32)
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    
    # Combine filtered strokes with noise
    enhanced = np.clip(filtered.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Convert back to PIL
    enhanced_img = Image.fromarray(enhanced)
    
    # Convert back to RGB if original was RGB
    if len(img_array.shape) == 3:
        enhanced_img = enhanced_img.convert('RGB')
    
    return enhanced_img


def create_stroke_mask(img):
    """
    Create a binary mask of strokes for attention guidance.
    
    Args:
        img: PIL Image (line drawing)
    
    Returns:
        Binary mask tensor (1 where strokes are, 0 elsewhere)
    """
    # Convert to grayscale
    if img.mode != 'L':
        gray = img.convert('L')
    else:
        gray = img
    
    # Convert to tensor and create mask
    gray_tensor = to_tensor(gray)
    # Invert so strokes are 1, background is 0
    stroke_mask = (gray_tensor < 0.9).float()
    
    return stroke_mask


def visualize_flow(flow, save_path=None):
    """
    Visualize optical flow as color-coded image.
    
    Args:
        flow: Flow tensor (B, 2, H, W)
        save_path: Optional path to save visualization
    
    Returns:
        PIL Image of flow visualization
    """
    # Convert to numpy
    flow_np = flow[0].cpu().numpy()  # (2, H, W)
    
    # Compute magnitude and angle
    magnitude = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
    angle = np.arctan2(flow_np[1], flow_np[0])
    
    # Normalize magnitude to [0, 1]
    magnitude = np.clip(magnitude / (magnitude.max() + 1e-8), 0, 1)
    
    # Convert to HSV
    h = (angle + np.pi) / (2 * np.pi)  # [0, 1]
    s = np.ones_like(magnitude)
    v = magnitude
    
    # Convert to RGB
    hsv = np.stack([h, s, v], axis=2)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Scale to [0, 255]
    rgb = (rgb * 255).astype(np.uint8)
    
    flow_img = Image.fromarray(rgb)
    
    if save_path:
        flow_img.save(save_path)
        print(f"✓ Saved flow visualization to: {save_path}")
    
    return flow_img


class SAINWrapperEnhanced:
    """
    Enhanced wrapper class for SAIN (Sketch-Aware Interpolation Network) model.
    Includes fixes for optical flow on line drawings and proper normalization.
    Normalized for the consolidated Animation AI backend.
    """
    
    def __init__(self, 
                 checkpoint_path: str = None,
                 device: Optional[str] = None,
                 size: int = 1024,  # Increased default size for better quality
                 enhance_strokes: bool = True,
                 dilation_radius: int = 2,
                 noise_amplitude: float = 0.05,
                 use_stroke_mask: bool = True,
                 debug_flow: bool = False):
        """
        Initialize the enhanced SAIN wrapper.
        
        Args:
            checkpoint_path: Path to the model checkpoint (defaults to consolidated backend path)
            device: Device to run the model on ('cuda', 'mps', 'cpu', or None for auto-detect)
            size: Input image size (max side, must be divisible by 8)
            enhance_strokes: Whether to enhance strokes for better optical flow
            dilation_radius: Radius for stroke dilation
            noise_amplitude: Amplitude of synthetic noise
            use_stroke_mask: Whether to use stroke mask for attention
            debug_flow: Whether to save flow visualizations for debugging
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
        self.enhance_strokes = enhance_strokes
        self.dilation_radius = dilation_radius
        self.noise_amplitude = noise_amplitude
        self.use_stroke_mask = use_stroke_mask
        self.debug_flow = debug_flow
        
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
        
        print(f"Enhanced SAIN wrapper initialized successfully on {self.device}!")
    
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
    
    def _preprocess_for_sain(self, image_path: str, target_w: int, target_h: int, pad: Tuple[int,int,int,int]) -> torch.Tensor:
        """
        Preprocess an image specifically for SAIN model (with proper normalization).
        
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
        tensor = to_tensor(img).unsqueeze(0)
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
        
        # Clamp to [0, 1] range
        tensor = tensor.clamp(0, 1)
        
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
            
            # Preprocess images for optical flow (with enhancement if enabled)
            if self.enhance_strokes:
                print("Enhancing strokes for optical flow...")
                img0_enhanced = enhance_strokes_for_flow(img0_orig, self.dilation_radius, self.noise_amplitude)
                img1_enhanced = enhance_strokes_for_flow(img1_orig, self.dilation_radius, self.noise_amplitude)
                
                # Resize and pad enhanced images
                img0_enhanced = img0_enhanced.resize((new_w, new_h), resample=Image.BICUBIC)
                img1_enhanced = img1_enhanced.resize((new_w, new_h), resample=Image.BICUBIC)
                img0_enhanced = pad_img(img0_enhanced, pad_left, pad_top, pad_right, pad_bottom)
                img1_enhanced = pad_img(img1_enhanced, pad_left, pad_top, pad_right, pad_bottom)
                
                # Convert to tensors for RAFT
                inp0 = to_tensor(img0_enhanced).unsqueeze(0).to(self.device)
                inp1 = to_tensor(img1_enhanced).unsqueeze(0).to(self.device)
            else:
                # Use original images for optical flow
                inp0 = to_tensor(pad_img(img0_orig.resize((new_w, new_h), resample=Image.BICUBIC), pad_left, pad_top, pad_right, pad_bottom)).unsqueeze(0).to(self.device)
                inp1 = to_tensor(pad_img(img1_orig.resize((new_w, new_h), resample=Image.BICUBIC), pad_left, pad_top, pad_right, pad_bottom)).unsqueeze(0).to(self.device)
            
            # Preprocess images for SAIN model (with proper normalization)
            img0_sain_tensor = self._preprocess_for_sain(img0_path, new_w, new_h, (pad_left, pad_top, pad_right, pad_bottom))
            img1_sain_tensor = self._preprocess_for_sain(img1_path, new_w, new_h, (pad_left, pad_top, pad_right, pad_bottom))
            
            # Create points tensor (attention mask)
            points = torch.ones((1, 1, target_h, target_w), device=self.device)
            
            # Compute optical flow
            print("Computing optical flow...")
            with torch.no_grad():
                flowAB = self.raft(inp0, inp1)[-1]
                flowBA = self.raft(inp1, inp0)[-1]
                region_flow = [flowAB, flowBA]
            
            # Save flow visualizations if debug is enabled
            if self.debug_flow:
                debug_dir = "debug_flow"
                os.makedirs(debug_dir, exist_ok=True)
                visualize_flow(flowAB, os.path.join(debug_dir, "flow_AB.png"))
                visualize_flow(flowBA, os.path.join(debug_dir, "flow_BA.png"))
            
            # Run SAIN model
            print("Running SAIN model...")
            with torch.no_grad():
                output = self.model(img0_sain_tensor, img1_sain_tensor, points, region_flow)
            
            # Postprocess output
            crop_box = (pad_left, pad_top, pad_left + new_w, pad_top + new_h)
            result = self._postprocess_tensor(output, crop_box)
            
            return result
            
        except Exception as e:
            print(f"Error during interpolation: {e}")
            import traceback
            traceback.print_exc()
            return None


def sain_enhanced_interpolate(img0_path: str,
                             img1_path: str,
                             t: float = 0.5,
                             checkpoint_path: str = None,
                             device: Optional[str] = None,
                             size: int = 1024,
                             enhance_strokes: bool = True,
                             dilation_radius: int = 2,
                             noise_amplitude: float = 0.05,
                             use_stroke_mask: bool = True,
                             debug_flow: bool = False) -> Optional[np.ndarray]:
    """
    Convenience function for one-shot enhanced SAIN interpolation.
    
    Args:
        img0_path: Path to the first input image
        img1_path: Path to the second input image
        t: Interpolation time (0.0 to 1.0)
        checkpoint_path: Path to the model checkpoint
        device: Device to run the model on
        size: Input image size
        enhance_strokes: Whether to enhance strokes for better optical flow
        dilation_radius: Radius for stroke dilation
        noise_amplitude: Amplitude of synthetic noise
        use_stroke_mask: Whether to use stroke mask for attention
        debug_flow: Whether to save flow visualizations for debugging
        
    Returns:
        Inbetween frame as numpy array, or None if failed
    """
    wrapper = SAINWrapperEnhanced(
        checkpoint_path=checkpoint_path, 
        device=device, 
        size=size,
        enhance_strokes=enhance_strokes,
        dilation_radius=dilation_radius,
        noise_amplitude=noise_amplitude,
        use_stroke_mask=use_stroke_mask,
        debug_flow=debug_flow
    )
    return wrapper.interpolate(img0_path, img1_path, t)


def main():
    """Example usage of the enhanced SAIN wrapper."""
    parser = argparse.ArgumentParser(description="Enhanced SAIN interpolation example")
    parser.add_argument('img0', help='Path to first input image')
    parser.add_argument('img1', help='Path to second input image')
    parser.add_argument('--t', type=float, default=0.5, help='Interpolation timestamp (0.0 to 1.0)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None, help='Device to use')
    parser.add_argument('--size', type=int, default=1024, help='Input image size (must be divisible by 8)')
    parser.add_argument('--no-enhance', action='store_true', help='Disable stroke enhancement')
    parser.add_argument('--debug-flow', action='store_true', help='Save flow visualizations')
    
    args = parser.parse_args()
    
    result = sain_enhanced_interpolate(
        args.img0, args.img1, args.t, 
        device=args.device, 
        size=args.size,
        enhance_strokes=not args.no_enhance,
        debug_flow=args.debug_flow
    )
    
    if result is not None:
        output_path = f"sain_enhanced_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(output_path, result)
        print(f"Interpolated frame saved to: {output_path}")
    else:
        print("Interpolation failed!")


if __name__ == "__main__":
    main() 