#!/usr/bin/env python3
"""
SAIN (Sketch-Aware Interpolation Network) CLI test script.
Normalized for the consolidated Animation AI backend.
"""

import os
import sys
import argparse
import cv2
from datetime import datetime
from pathlib import Path

# Add the wrappers directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'wrappers'))

from sain_wrapper import sain_interpolate


def main():
    parser = argparse.ArgumentParser(description="SAIN interpolation test")
    parser.add_argument('img0', help='Path to first input image')
    parser.add_argument('img1', help='Path to second input image')
    parser.add_argument('--t', type=float, default=0.5, help='Interpolation timestamp (0.0 to 1.0)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None, help='Device to use')
    parser.add_argument('--size', type=int, default=512, help='Input image size (must be divisible by 8)')
    parser.add_argument('--gif', action='store_true', help='Save output as GIF instead of PNG')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.img0):
        print(f"Error: First image not found: {args.img0}")
        return 1
    
    if not os.path.exists(args.img1):
        print(f"Error: Second image not found: {args.img1}")
        return 1
    
    if args.t < 0.0 or args.t > 1.0:
        print(f"Error: Interpolation timestamp must be between 0.0 and 1.0, got {args.t}")
        return 1
    
    if args.size % 8 != 0:
        print(f"Error: Size must be divisible by 8, got {args.size}")
        return 1
    
    print(f"SAIN Interpolation Test")
    print(f"=======================")
    print(f"Input 0: {args.img0}")
    print(f"Input 1: {args.img1}")
    print(f"Timestamp: {args.t}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Size: {args.size}")
    print(f"Output format: {'gif' if args.gif else 'png'}")
    print()
    
    # Create output directory
    output_dir = os.path.join(current_dir, 'test_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = 'sain'
    ext = 'gif' if args.gif else 'png'
    output_path = os.path.join(output_dir, f"{timestamp}_{model_name}_test.{ext}")
    
    try:
        print("Running SAIN interpolation...")
        result = sain_interpolate(
            img0_path=args.img0,
            img1_path=args.img1,
            t=args.t,
            device=args.device,
            size=args.size
        )
        
        if result is not None:
            # Always save PNG first
            png_path = output_path.replace('.gif', '.png') if args.gif else output_path
            cv2.imwrite(png_path, result)
            print(f"✓ Interpolated frame saved to: {png_path}")
            
            if args.gif:
                # Create GIF with input frames and interpolated frame
                import imageio
                
                # Get the dimensions of the interpolated result
                result_height, result_width = result.shape[:2]
                
                # Load and resize input images to match result dimensions
                img0 = cv2.imread(args.img0)
                img1 = cv2.imread(args.img1)
                img0_resized = cv2.resize(img0, (result_width, result_height))
                img1_resized = cv2.resize(img1, (result_width, result_height))
                
                # Convert BGR to RGB for imageio
                img0_rgb = cv2.cvtColor(img0_resized, cv2.COLOR_BGR2RGB)
                img1_rgb = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                
                # Save high-quality GIF with better settings
                imageio.mimsave(output_path, [img0_rgb, result_rgb, img1_rgb], 
                              duration=0.5, 
                              optimize=False,  # Disable optimization for better quality
                              quality=10)  # Higher quality (lower compression)
                print(f"✓ GIF saved to: {output_path}")
            
            print(f"✓ SAIN interpolation completed successfully!")
            return 0
        else:
            print("✗ SAIN interpolation failed!")
            return 1
    except Exception as e:
        print(f"✗ Error during SAIN interpolation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 