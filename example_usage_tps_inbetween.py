#!/usr/bin/env python3
"""
Example usage script for TPS-Inbetween wrapper in the consolidated Animation AI backend.

This script demonstrates the normalized interface for generating inbetween frames
between two input images with configurable parameters.

Usage:
    python3 example_usage_tps_inbetween.py --img0 input_images/img1.jpg --img1 input_images/img2.jpg --device mps --size 720 --gif
"""

import os
import sys
import argparse
from datetime import datetime
from wrappers.tps_inbetween_wrapper import TPSInbetweenWrapper


def main():
    """Generate inbetween frames using TPS-Inbetween with normalized interface."""
    parser = argparse.ArgumentParser(
        description="Generate inbetween frames using TPS-Inbetween model",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--img0', required=True, 
                       help='Path to the first input image')
    parser.add_argument('--img1', required=True, 
                       help='Path to the second input image')
    
    # Optional arguments with defaults
    parser.add_argument('--size', type=int, default=720,
                       help='Maximum image dimension for resizing (default: 720)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None,
                       help='Device to use for computation (auto-detect if not specified)')
    parser.add_argument('--gif', action='store_true',
                       help='Create GIF output with img0 + inbetween + img1')
    parser.add_argument('--gif_duration', type=float, default=0.05,
                       help='Duration per frame in GIF (default: 0.05 seconds)')
    parser.add_argument('--num_frames', type=int, default=1,
                       help='Number of intermediate frames to generate (default: 1)')
    parser.add_argument('--output_dir', default='test_outputs',
                       help='Output directory (default: test_outputs)')
    parser.add_argument('--temp_dir', default='temp',
                       help='Temporary directory (default: temp)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.img0):
        print(f"Error: Input image 0 not found: {args.img0}")
        sys.exit(1)
    
    if not os.path.exists(args.img1):
        print(f"Error: Input image 1 not found: {args.img1}")
        sys.exit(1)
    
    # Create output and temp directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Generate timestamped output filename with unified naming convention
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = 'tps_inbetween'
    ext = 'gif' if args.gif else 'png'
    output_filename = f"{timestamp}_{model_name}_test.{ext}"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print("=" * 60)
    print("TPS-Inbetween Inbetween Frame Generation")
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
        # Initialize the wrapper
        print("\nInitializing TPS-Inbetween wrapper...")
        wrapper = TPSInbetweenWrapper(device=args.device)
        
        # Generate inbetween frame(s)
        print(f"\nGenerating inbetween frame(s)...")
        result_path = wrapper.interpolate(
            img0_path=args.img0,
            img1_path=args.img1,
            output_path=output_path,
            num_frames=args.num_frames,
            temp_dir=args.temp_dir,
            max_image_size=args.size,
            create_gif=args.gif,
            gif_duration=args.gif_duration
        )
        
        if result_path:
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            
            # Always save PNG first
            png_path = output_path.replace('.gif', '.png') if args.gif else output_path
            if args.gif and not os.path.exists(png_path):
                # If GIF was created but PNG doesn't exist, create it from the GIF
                import cv2
                import imageio
                gif_frames = imageio.mimread(output_path)
                if gif_frames:
                    # Save the middle frame (inbetween) as PNG
                    middle_frame = gif_frames[1]  # Index 1 is the inbetween frame
                    cv2.imwrite(png_path, cv2.cvtColor(middle_frame, cv2.COLOR_RGB2BGR))
                    print(f"✓ Interpolated frame saved to: {png_path}")
            
            print(f"Inbetween frame saved to: {result_path}")
            
            if args.gif:
                print(f"✓ GIF saved to: {output_path}")
            
            print(f"Timestamp: {timestamp}")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("ERROR: Failed to generate inbetween frame!")
            print("=" * 60)
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the input images exist at the specified paths.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 