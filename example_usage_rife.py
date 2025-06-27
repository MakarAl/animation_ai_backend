#!/usr/bin/env python3
"""
Example usage script for RIFE wrapper in the consolidated Animation AI backend.

Usage:
    python3 example_usage_rife.py --img0 input_images/img1.jpg --img1 input_images/img2.jpg --device mps --size 720 --gif
"""

import os
import sys
import argparse
from datetime import datetime
import cv2
import imageio
from wrappers.rife_wrapper import RIFEInterpolator


def main():
    parser = argparse.ArgumentParser(
        description="Generate inbetween frames using RIFE model",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--img0', required=True, help='Path to the first input image')
    parser.add_argument('--img1', required=True, help='Path to the second input image')
    parser.add_argument('--size', type=int, default=720, help='Maximum image dimension for resizing (default: 720)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None, help='Device to use for computation (auto-detect if not specified)')
    parser.add_argument('--gif', action='store_true', help='Create GIF output with img0 + inbetween + img1')
    parser.add_argument('--num_frames', type=int, default=1, help='Number of intermediate frames to generate (default: 1)')
    parser.add_argument('--output_dir', default='test_outputs', help='Output directory (default: test_outputs)')
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.img0):
        print(f"Error: Input image 0 not found: {args.img0}")
        sys.exit(1)
    if not os.path.exists(args.img1):
        print(f"Error: Input image 1 not found: {args.img1}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = 'rife'
    ext = 'gif' if args.gif else 'png'
    output_filename = f"{timestamp}_{model_name}_test.{ext}"
    output_path = os.path.join(args.output_dir, output_filename)

    print("=" * 60)
    print("RIFE Inbetween Frame Generation")
    print("=" * 60)
    print(f"Input 0: {args.img0}")
    print(f"Input 1: {args.img1}")
    print(f"Output: {output_path}")
    print(f"Max size: {args.size}")
    print(f"Device: {args.device or 'auto-detect'}")
    print(f"Create GIF: {args.gif}")
    print(f"Number of frames: {args.num_frames}")
    print("=" * 60)

    try:
        print("\nInitializing RIFE wrapper...")
        wrapper = RIFEInterpolator(device=args.device)
        print("\nGenerating inbetween frame(s)...")
        result_imgs = wrapper.interpolate(
            img0=args.img0,
            img1=args.img1,
            exp=args.num_frames,
            max_size=args.size
        )
        
        # Get the middle frame (or first intermediate if only one)
        if len(result_imgs) == 3:
            out_img = result_imgs[1]
        elif len(result_imgs) > 1:
            out_img = result_imgs[len(result_imgs)//2]
        else:
            out_img = result_imgs[0]
        
        # Always save PNG first
        png_path = output_path.replace('.gif', '.png') if args.gif else output_path
        cv2.imwrite(png_path, out_img)
        print(f"✓ Interpolated frame saved to: {png_path}")
        
        if args.gif:
            # Save as GIF (img0, inbetweens, img1)
            imageio.mimsave(output_path, [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in result_imgs], 
                          duration=0.1, 
                          optimize=False,  # Disable optimization for better quality
                          quality=10)  # Higher quality (lower compression)
            print(f"✓ GIF saved to: {output_path}")
        
        print(f"\nSUCCESS!")
        print(f"Timestamp: {timestamp}")
        print("=" * 60)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 