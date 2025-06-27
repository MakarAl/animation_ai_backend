import os
import sys
import cv2
import torch
import numpy as np
from torch.nn import functional as F
import warnings
from typing import Union, List, Optional
from pathlib import Path
import tempfile
import shutil

warnings.filterwarnings("ignore")

# Set up model directory for consolidated backend
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(current_dir), 'models', 'RIFE')
# Patch sys.path for model imports
sys.path.insert(0, models_dir)  # Add models/RIFE/ so 'model' and 'train_log' are importable

class RIFEInterpolator:
    """
    A wrapper class for RIFE (Real-time Intermediate Flow Estimation) image interpolation.
    Normalized for the consolidated Animation AI backend.
    """
    def __init__(self, model_dir: str = None, device: Optional[str] = None):
        """
        Initialize the RIFE interpolator.
        Args:
            model_dir: Directory containing the trained model files (auto-detected if None)
            device: Device to run inference on ('cuda', 'cpu', 'mps', or None for auto-detection)
        """
        if model_dir is None:
            model_dir = os.path.join(models_dir, 'train_log')
        self.model_dir = model_dir
        # Device selection logic
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        self.device_torch = torch.device(self.device)
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        # Load model
        self.model = self._load_model()
        self.model.eval()
        self.model.device()

    def _load_model(self):
        """Load the appropriate RIFE model based on available files."""
        try:
            # Try HDv3 model first (most recent)
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(self.model_dir, -1)
            print("Loaded v3.x HD model.")
        except ImportError:
            try:
                # Try base RIFE model
                from model.RIFE import Model
                model = Model()
                model.load_model(self.model_dir, -1)
                print("Loaded base RIFE model.")
            except ImportError as e:
                print(f"Error loading RIFE models: {e}")
                print("Available models:")
                print(f"  - {os.path.join(models_dir, 'model/RIFE.py')}")
                print(f"  - {os.path.join(models_dir, 'train_log/RIFE_HDv3.py')}")
                raise
        return model

    def _load_image(self, image_path: Union[str, np.ndarray], max_size: int = 720) -> torch.Tensor:
        """
        Load and preprocess an image, resizing to max_size while preserving aspect ratio.
        Args:
            image_path: Path to image file or numpy array
            max_size: Maximum image dimension
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
        else:
            img = image_path
        # Resize to max_size while preserving aspect ratio
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            if h > w:
                new_h, new_w = max_size, int(w * max_size / h)
            else:
                new_w, new_h = max_size, int(h * max_size / w)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_tensor = (torch.tensor(img.transpose(2, 0, 1)).to(self.device_torch) / 255.)
        return img_tensor.unsqueeze(0)

    def _pad_image(self, img: torch.Tensor) -> tuple:
        n, c, h, w = img.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        padded_img = F.pad(img, padding)
        return padded_img, h, w

    def _tensor_to_numpy(self, img_tensor: torch.Tensor, h: int, w: int) -> np.ndarray:
        return (img_tensor[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    def interpolate(self, img0: Union[str, np.ndarray], img1: Union[str, np.ndarray],
                    exp: int = 1, ratio: Optional[float] = None, max_size: int = 720) -> List[np.ndarray]:
        # Load and resize images
        img0_tensor = self._load_image(img0, max_size=max_size)
        img1_tensor = self._load_image(img1, max_size=max_size)
        # Pad images
        img0_padded, h, w = self._pad_image(img0_tensor)
        img1_padded, _, _ = self._pad_image(img1_tensor)
        # Perform interpolation
        if ratio is not None:
            img_list = [img0_padded]
            img0_ratio = 0.0
            img1_ratio = 1.0
            rthreshold = 0.02
            rmaxcycles = 8
            if ratio <= img0_ratio + rthreshold / 2:
                middle = img0_padded
            elif ratio >= img1_ratio - rthreshold / 2:
                middle = img1_padded
            else:
                tmp_img0 = img0_padded
                tmp_img1 = img1_padded
                for inference_cycle in range(rmaxcycles):
                    middle = self.model.inference(tmp_img0, tmp_img1)
                    middle_ratio = (img0_ratio + img1_ratio) / 2
                    if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                        break
                    if ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img1_padded)
        else:
            img_list = [img0_padded, img1_padded]
            for i in range(exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = self.model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1_padded)
                img_list = tmp
        # Convert back to numpy arrays
        result = []
        for img_tensor in img_list:
            img_np = self._tensor_to_numpy(img_tensor, h, w)
            result.append(img_np)
        return result

    def interpolate_and_save(self, img0: Union[str, np.ndarray], img1: Union[str, np.ndarray],
                            output_dir: str = 'output', exp: int = 1, ratio: Optional[float] = None,
                            max_size: int = 720) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        interpolated_images = self.interpolate(img0, img1, exp, ratio, max_size)
        saved_paths = []
        for idx, img in enumerate(interpolated_images):
            out_path = os.path.join(output_dir, f"frame_{idx:02d}.png")
            cv2.imwrite(out_path, img)
            saved_paths.append(out_path)
        return saved_paths 