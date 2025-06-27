import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision
from datetime import datetime
import imageio
import copy
import argparse

# Add the models directory to the path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(current_dir), 'models', 'TPS_Inbetween')
sys.path.append(models_dir)

from model.gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from model.gluestick.models.two_view_pipeline import TwoViewPipeline
from model.tpsinbet import TPS_inbet
from util.utils import batch_dog


class TPSInbetweenWrapper:
    """
    A wrapper class for the TPS-Inbetween model that provides a simple interface
    for generating inbetween frames between two input images.
    
    Normalized interface for the consolidated Animation AI backend.
    """
    
    def __init__(self, model_path=None, device=None, use_cpu=False):
        """
        Initialize the TPS-Inbetween wrapper.
        
        Args:
            model_path (str): Path to the pretrained model weights (auto-detected if None)
            device (str): Device to run the model on ('cuda', 'mps', or 'cpu')
            use_cpu (bool): Force CPU usage if True
        """
        # Auto-detect model path if not provided
        if model_path is None:
            model_path = os.path.join(models_dir, 'ckpt', 'model_latest.pt')
        
        # Device selection logic
        if use_cpu:
            self.device = 'cpu'
        elif device is not None:
            self.device = device
        else:
            # Auto-detect best available device
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        
        self.model_path = model_path
        self.matching_model = None
        self.inbetween_model = None
        
        print(f"TPS-Inbetween wrapper initialized with device: {self.device}")
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load the matching model and inbetween model."""
        print("Loading matching model (GlueStick)...")
        
        # Update GlueStick weights path to use models directory
        gluestick_weights = os.path.join(models_dir, 'model', 'resources', 'weights', 'checkpoint_GlueStick_MD.tar')
        
        # Matching model configuration
        conf = {
            'name': 'two_view_pipeline',
            'use_lines': True,
            'extractor': {
                'name': 'wireframe',
                'sp_params': {
                    'force_num_keypoints': False,
                    'max_num_keypoints': 1000,
                },
                'wireframe_params': {
                    'merge_points': True,
                    'merge_line_endpoints': True,
                },
                'max_n_lines': 300,
            },
            'matcher': {
                'name': 'gluestick',
                'weights': gluestick_weights,
                'trainable': False,
            },
            'ground_truth': {
                'from_pose_depth': False,
            }
        }
        
        self.matching_model = TwoViewPipeline(conf).to(self.device).eval()
        
        print("Loading TPS-Inbetween model...")
        # Create args object for the model
        class Args:
            def __init__(self, device):
                self.cpu = device == 'cpu'
                self.xN = 1  # Default to 1 intermediate frame
                self.device = device
        
        args = Args(self.device)
        
        self.inbetween_model = TPS_inbet(args)
        self.inbetween_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.inbetween_model = self.inbetween_model.to(self.device).eval()
        
        print("Models loaded successfully!")
    
    def _img_open_torch(self, img_path, max_size=512, gray=True):
        """Load and preprocess image for torch. Aspect-ratio resize to fit max_size."""
        img = Image.open(img_path)
        w, h = img.size
        if max(w, h) > max_size:
            if w > h:
                new_w, new_h = max_size, int(h * max_size / w)
            else:
                new_w, new_h = int(w * max_size / h), max_size
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f"Resized image from {w}x{h} to {new_w}x{new_h} to prevent memory issues")
        if gray:
            img = img.convert('L')
            np_img = np.array(img)[..., None]
        else:
            np_img = np.array(img)
        np_img = np_img.astype('float64')
        np_transpose = np.ascontiguousarray(np_img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(1.0 / 255)
        return tensor.unsqueeze(0), img

    def _center_crop(self, img, target_w, target_h):
        """Center crop a PIL image to (target_w, target_h)."""
        w, h = img.size
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        return img.crop((left, top, right, bottom))

    def _norm_flow(self, flow, h, w):
        """Normalize flow coordinates."""
        flow_norm = flow.copy()
        flow_norm[:, 0] = flow_norm[:, 0] / w
        flow_norm[:, 1] = flow_norm[:, 1] / h
        return flow_norm
    
    def _generate_matches(self, img0_path, img1_path, temp_dir='./temp', max_image_size=512):
        os.makedirs(temp_dir, exist_ok=True)
        # Load and aspect-ratio resize both images
        tensor0, pil0 = self._img_open_torch(img0_path, max_size=max_image_size)
        tensor1, pil1 = self._img_open_torch(img1_path, max_size=max_image_size)
        # Find minimum common size
        w0, h0 = pil0.size
        w1, h1 = pil1.size
        target_w = min(w0, w1)
        target_h = min(h0, h1)
        # Center crop both images to the same size
        pil0_cropped = self._center_crop(pil0, target_w, target_h)
        pil1_cropped = self._center_crop(pil1, target_w, target_h)
        # Convert back to tensors
        def pil_to_tensor(img):
            np_img = np.array(img)[..., None]
            np_img = np_img.astype('float64')
            np_transpose = np.ascontiguousarray(np_img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(1.0 / 255)
            return tensor.unsqueeze(0)
        torch_gray0 = pil_to_tensor(pil0_cropped).to(self.device)
        torch_gray1 = pil_to_tensor(pil1_cropped).to(self.device)
        b, c, h, w = torch_gray0.shape
        x = {'image0': torch_gray0, 'image1': torch_gray1}
        pred = self.matching_model(x)
        pred = batch_to_np(pred)
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0 = pred["matches0"]
        valid_matches = m0 != -1
        match_indices = m0[valid_matches]
        matched_kps0 = kp0[valid_matches]
        matched_kps1 = kp1[match_indices]
        n_kps0 = self._norm_flow(matched_kps0, h, w)
        n_kps1 = self._norm_flow(matched_kps1, h, w)
        kps_stack = np.stack((n_kps0, n_kps1), axis=0)
        matches_path = os.path.join(temp_dir, 'matches.npy')
        np.save(matches_path, kps_stack)
        return matches_path, torch_gray0, torch_gray1
    
    def interpolate(self, img0_path, img1_path, output_path=None, num_frames=1, 
                   save_intermediate=True, temp_dir='./temp', max_image_size=512, 
                   create_gif=False, gif_duration=0.05):
        """
        Generate inbetween frames between two input images.
        
        Args:
            img0_path (str): Path to the first input image
            img1_path (str): Path to the second input image
            output_path (str): Path to save the output image (if None, auto-generate)
            num_frames (int): Number of intermediate frames to generate
            save_intermediate (bool): Whether to save intermediate frames
            temp_dir (str): Directory for temporary files
            max_image_size (int): Maximum image dimension to prevent memory issues
            create_gif (bool): Whether to create a GIF with input0, inbetween, input1
            gif_duration (float): Duration per frame in the GIF
            
        Returns:
            str: Path to the saved output image
        """
        if not os.path.exists(img0_path):
            raise FileNotFoundError(f"Input image 0 not found: {img0_path}")
        if not os.path.exists(img1_path):
            raise FileNotFoundError(f"Input image 1 not found: {img1_path}")
        
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate matches
        print("Generating matches between images...")
        matches_path, torch_gray0, torch_gray1 = self._generate_matches(img0_path, img1_path, temp_dir, max_image_size)
        
        # Update model to generate requested number of frames
        # Note: The model expects xN+1 total frames, so we need num_frames+1
        # But we need to handle the case where num_frames=1 specially
        if num_frames == 1:
            # For single frame, use t=0.5 (middle)
            self.inbetween_model.times = torch.tensor([0.5]).to(self.device)
        else:
            self.inbetween_model.times = torch.linspace(0, 1, num_frames + 1)[1:-1].to(self.device)
        
        # Generate inbetween frames
        print(f"Generating {num_frames} inbetween frames...")
        with torch.no_grad():
            pred, _ = self.inbetween_model(1 - torch_gray0, 1 - torch_gray1, [matches_path])
        
        pred = [1 - p for p in pred]
        
        # Verify we got the expected number of frames
        if len(pred) != num_frames:
            print(f"Warning: Expected {num_frames} frames, got {len(pred)}")
            if len(pred) == 0:
                print("Error: No frames generated! This might indicate a model issue.")
                return None
        
        # Save results
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"tps_inbetween_{timestamp}"
            output_path = os.path.join(temp_dir, f"{base_name}.png")
        
        # Save the middle frame (or first intermediate frame if only one)
        # Use the middle frame for better interpolation
        if len(pred) > 0:
            if len(pred) == 1:
                output_frame = pred[0]  # Single intermediate frame
            else:
                # Use the middle frame for better interpolation
                middle_idx = len(pred) // 2
                output_frame = pred[middle_idx]
        else:
            print("Error: No intermediate frames generated!")
            return None
        
        # Convert to PIL and save
        output_img = torchvision.transforms.ToPILImage()(output_frame.float().squeeze(0))
        output_img.save(output_path)
        
        # Save intermediate frames if requested
        if save_intermediate and len(pred) > 1:
            intermediate_dir = os.path.join(temp_dir, 'intermediate_frames')
            os.makedirs(intermediate_dir, exist_ok=True)
            
            for idx, frame in enumerate(pred):
                frame_path = os.path.join(intermediate_dir, f'frame_{idx+1}.png')
                torchvision.transforms.ToPILImage()(frame.float().squeeze(0)).save(frame_path)
        
        # Create GIF if requested
        gif_path = None
        if create_gif:
            gif_path = self._create_simple_gif(torch_gray0, pred, torch_gray1, output_path, gif_duration)
        
        print(f"Inbetween frame saved to: {output_path}")
        if gif_path:
            print(f"GIF saved to: {gif_path}")
        return output_path
    
    def _create_simple_gif(self, torch_gray0, pred, torch_gray1, output_path, duration=0.05):
        """Create a simple GIF with input0, inbetween, input1."""
        # Create frames directory
        frames_dir = os.path.dirname(output_path)
        frames_dir = os.path.join(frames_dir, 'gif_frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save all frames
        frame_paths = []
        
        # Save input frame 0
        frame0_path = os.path.join(frames_dir, 'frame_0.png')
        torchvision.transforms.ToPILImage()(torch_gray0.float().squeeze(0)).save(frame0_path)
        frame_paths.append(frame0_path)
        
        # Save intermediate frames
        for idx, frame in enumerate(pred):
            frame_path = os.path.join(frames_dir, f'frame_{idx+1}.png')
            torchvision.transforms.ToPILImage()(frame.float().squeeze(0)).save(frame_path)
            frame_paths.append(frame_path)
        
        # Save input frame 1
        frame1_path = os.path.join(frames_dir, f'frame_{len(pred)+1}.png')
        torchvision.transforms.ToPILImage()(torch_gray1.float().squeeze(0)).save(frame1_path)
        frame_paths.append(frame1_path)
        
        # Create GIF
        gif_path = output_path.replace('.png', '.gif')
        frames = []
        for frame_path in frame_paths:
            frames.append(imageio.v2.imread(frame_path))
        imageio.mimsave(gif_path, frames, duration=duration)
        
        return gif_path
    
    def interpolate_sequence(self, img0_path, img1_path, output_dir=None, num_frames=30, 
                           temp_dir='./temp'):
        """
        Generate a sequence of inbetween frames and save as GIF.
        
        Args:
            img0_path (str): Path to the first input image
            img1_path (str): Path to the second input image
            output_dir (str): Directory to save output files
            num_frames (int): Number of intermediate frames to generate
            temp_dir (str): Directory for temporary files
            
        Returns:
            str: Path to the saved GIF file
        """
        if output_dir is None:
            output_dir = temp_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate matches
        print("Generating matches between images...")
        matches_path, torch_gray0, torch_gray1 = self._generate_matches(img0_path, img1_path, temp_dir)
        
        # Update model to generate requested number of frames
        self.inbetween_model.times = torch.linspace(0, 1, num_frames + 1)[1:-1].to(self.device)
        
        # Generate inbetween frames
        print(f"Generating {num_frames} inbetween frames...")
        with torch.no_grad():
            pred, _ = self.inbetween_model(1 - torch_gray0, 1 - torch_gray1, [matches_path])
        
        pred = [1 - p for p in pred]
        
        # Save individual frames
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save input frames
        torchvision.transforms.ToPILImage()(torch_gray0.float().squeeze(0)).save(
            os.path.join(frames_dir, '0.png'))
        torchvision.transforms.ToPILImage()(torch_gray1.float().squeeze(0)).save(
            os.path.join(frames_dir, f'{len(pred)+1}.png'))
        
        # Save intermediate frames
        for idx, frame in enumerate(pred):
            frame_path = os.path.join(frames_dir, f'{idx+1}.png')
            torchvision.transforms.ToPILImage()(frame.float().squeeze(0)).save(frame_path)
        
        # Create GIF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(output_dir, f'tps_sequence_{timestamp}.gif')
        
        # Collect all frame paths
        frame_paths = [os.path.join(frames_dir, f'{i}.png') for i in range(num_frames + 1)]
        
        # Create GIF
        frames = []
        for frame_path in frame_paths:
            frames.append(imageio.v2.imread(frame_path))
        imageio.mimsave(gif_path, frames, duration=0.05)
        
        print(f"Sequence saved to: {gif_path}")
        return gif_path


def main():
    """Example usage of the TPS-Inbetween wrapper."""
    parser = argparse.ArgumentParser(description='TPS-Inbetween Wrapper')
    parser.add_argument('--img0', required=True, help='Path to first input image')
    parser.add_argument('--img1', required=True, help='Path to second input image')
    parser.add_argument('--output', help='Output path (auto-generated if not specified)')
    parser.add_argument('--num_frames', type=int, default=1, help='Number of intermediate frames')
    parser.add_argument('--sequence', action='store_true', help='Generate full sequence as GIF')
    parser.add_argument('--model_path', help='Path to model weights (auto-detected if not specified)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU only')
    parser.add_argument('--temp_dir', default='./temp', help='Temporary directory')
    
    args = parser.parse_args()
    
    # Initialize wrapper
    wrapper = TPSInbetweenWrapper(
        model_path=args.model_path,
        use_cpu=args.cpu
    )
    
    if args.sequence:
        # Generate full sequence
        output_path = wrapper.interpolate_sequence(
            args.img0, args.img1, 
            output_dir=args.output,
            num_frames=args.num_frames,
            temp_dir=args.temp_dir
        )
    else:
        # Generate single inbetween frame
        output_path = wrapper.interpolate(
            args.img0, args.img1,
            output_path=args.output,
            num_frames=args.num_frames,
            temp_dir=args.temp_dir
        )
    
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main() 