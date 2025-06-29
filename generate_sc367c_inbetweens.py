#!/usr/bin/env python3
"""
Script to generate inbetweens for ep01_sc367c_Cyberslav_CLEAN_ frames using multiple models.

This script processes consecutive pairs of keyframes and generates inbetween frames
with the following models and configurations:
- TPS-Inbetween (with/without vector cleanup)
- RIFE
- SAIN (basic and enhanced)

All experiments use:
- device = cpu
- size = 1440
- no_edge_sharpen = true
- no uniform_thin = true

The script creates individual inbetweens and final combined GIFs for each model+setup.
"""

import os
import sys
import glob
import time
from datetime import datetime
from pathlib import Path
import imageio
from PIL import Image
import numpy as np

# Patch sys.path for RIFE/SAIN model imports
import sys, os
backend_dir = os.path.dirname(os.path.abspath(__file__))
rife_dir = os.path.join(backend_dir, 'models', 'RIFE')
rife_model_dir = os.path.join(rife_dir, 'model')
rife_train_log_dir = os.path.join(rife_dir, 'train_log')
for p in [backend_dir, rife_dir, rife_model_dir, rife_train_log_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Add the current directory to the path to import the wrappers
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import wrappers directly - no error handling to hide real issues
from wrappers.tps_inbetween_wrapper import TPSInbetweenWrapper
from wrappers.rife_wrapper import RIFEInterpolator
from wrappers.sain_wrapper import SAINWrapper
from wrappers.sain_enhanced_wrapper import SAINWrapperEnhanced

print("✓ All wrappers imported successfully")


def get_frame_paths(cleaned_images_dir, frame_numbers):
    """
    Get the full paths for the specified frame numbers.
    
    Args:
        cleaned_images_dir (str): Path to the cleaned_images directory
        frame_numbers (list): List of frame numbers as strings
    
    Returns:
        list: List of full file paths
    """
    frame_paths = []
    for frame_num in frame_numbers:
        # Pad frame number to 4 digits
        padded_frame = frame_num.zfill(4)
        filename = f"ep01_sc367c_Cyberslav_CLEAN_{padded_frame}.jpg"
        filepath = os.path.join(cleaned_images_dir, filename)
        
        if os.path.exists(filepath):
            frame_paths.append(filepath)
        else:
            print(f"Warning: Frame {filename} not found at {filepath}")
    
    return frame_paths


def create_combined_gif(frame_paths, output_path, duration=0.1):
    """
    Create a combined GIF with all frames in sequence.
    
    Args:
        frame_paths (list): List of paths to all frames (keyframes + inbetweens)
        output_path (str): Path to save the combined GIF
        duration (float): Duration per frame in seconds
    """
    print(f"Creating combined GIF: {output_path}")
    
    # Load all frames
    frames = []
    target_size = None
    
    for frame_path in frame_paths:
        if os.path.exists(frame_path):
            # Load image and convert to RGB if needed
            img = Image.open(frame_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Determine target size from first frame
            if target_size is None:
                target_size = img.size
            else:
                # Resize to match target size
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            frames.append(np.array(img))
        else:
            print(f"Warning: Frame not found: {frame_path}")
    
    if frames:
        # Save as GIF
        imageio.mimsave(output_path, frames, duration=duration)
        print(f"✓ Combined GIF saved: {output_path}")
        print(f"  - {len(frames)} frames")
        print(f"  - Frame size: {target_size}")
        print(f"  - Duration per frame: {duration}s")
    else:
        print("Error: No frames to combine!")


def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.2f}s"


def run_experiment(experiment_name, wrapper, frame_paths, frame_numbers, output_dir, temp_dir, size):
    """
    Run a single experiment with the given wrapper and configuration.
    
    Args:
        experiment_name (str): Name of the experiment
        wrapper: The wrapper instance to use
        frame_paths (list): List of frame paths
        frame_numbers (list): List of frame numbers
        output_dir (str): Output directory for this experiment
        temp_dir (str): Temporary directory
        size (int): Maximum image size
    
    Returns:
        dict: Experiment results with timing and file paths
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    
    # Create experiment-specific directories
    experiment_output_dir = os.path.join(output_dir, experiment_name)
    experiment_temp_dir = os.path.join(temp_dir, experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)
    os.makedirs(experiment_temp_dir, exist_ok=True)
    
    # Initialize wrapper timing
    wrapper_init_start = time.time()
    print(f"Initializing {experiment_name} wrapper...")
    # Note: Wrapper is already initialized, just timing the process
    wrapper_init_time = time.time() - wrapper_init_start
    print(f"✓ Wrapper ready in {format_time(wrapper_init_time)}")
    
    # Generate inbetweens for consecutive pairs
    all_frame_paths = []  # Will contain all frames in order for final GIF
    inbetween_results = []
    frame_generation_times = []
    
    for i in range(len(frame_paths) - 1):
        img0_path = frame_paths[i]
        img1_path = frame_paths[i + 1]
        
        # Extract frame numbers for naming
        frame0_num = frame_numbers[i]
        frame1_num = frame_numbers[i + 1]
        
        print(f"\n{'='*60}")
        print(f"Processing pair {i+1}/{len(frame_paths)-1}: {frame0_num} → {frame1_num}")
        print(f"{'='*60}")
        
        # Generate timestamp for this inbetween
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output filename
        output_filename = f"{timestamp}_{frame0_num}_to_{frame1_num}"
        output_path = os.path.join(experiment_output_dir, f"{output_filename}.png")
        gif_path = os.path.join(experiment_output_dir, f"{output_filename}.gif")
        
        print(f"Input 0: {os.path.basename(img0_path)}")
        print(f"Input 1: {os.path.basename(img1_path)}")
        print(f"Output PNG: {output_filename}.png")
        print(f"Output GIF: {output_filename}.gif")
        
        # Start timing for this frame pair
        frame_start_time = time.time()
        
        try:
            # Generate inbetween based on wrapper type
            if isinstance(wrapper, TPSInbetweenWrapper):
                result_path = wrapper.interpolate(
                    img0_path=img0_path,
                    img1_path=img1_path,
                    output_path=output_path,
                    num_frames=1,
                    temp_dir=experiment_temp_dir,
                    max_image_size=size,
                    create_gif=True,
                    gif_duration=0.05
                )
            elif isinstance(wrapper, RIFEInterpolator):
                # RIFE returns a list of numpy arrays, we want the middle one
                interpolated_frames = wrapper.interpolate(
                    img0=img0_path,
                    img1=img1_path,
                    exp=1,  # Generate 1 intermediate frame
                    max_size=size
                )
                # Save the middle frame (index 1) as PNG
                import cv2
                middle_frame = interpolated_frames[1]  # Index 1 is the inbetween
                cv2.imwrite(output_path, middle_frame)
                result_path = output_path
                
                # Create GIF manually
                gif_frames = []
                for frame in interpolated_frames:
                    gif_frames.append(frame)
                imageio.mimsave(gif_path, gif_frames, duration=0.05)
                
            elif isinstance(wrapper, (SAINWrapper, SAINWrapperEnhanced)):
                # SAIN returns a numpy array
                result_array = wrapper.interpolate(
                    img0_path=img0_path,
                    img1_path=img1_path,
                    t=0.5  # Middle frame
                )
                
                if result_array is not None:
                    # Save the result as PNG
                    import cv2
                    cv2.imwrite(output_path, result_array)
                    result_path = output_path
                    
                    # Create GIF manually with input0, inbetween, input1
                    import cv2
                    img0 = cv2.imread(img0_path)
                    img1 = cv2.imread(img1_path)
                    
                    # Resize all to same size for GIF
                    target_size = (result_array.shape[1], result_array.shape[0])
                    img0_resized = cv2.resize(img0, target_size)
                    img1_resized = cv2.resize(img1, target_size)
                    
                    gif_frames = [img0_resized, result_array, img1_resized]
                    imageio.mimsave(gif_path, gif_frames, duration=0.05)
                else:
                    result_path = None
            else:
                raise ValueError(f"Unknown wrapper type: {type(wrapper)}")
            
            # Calculate generation time for this frame
            frame_generation_time = time.time() - frame_start_time
            frame_generation_times.append(frame_generation_time)
            
            if result_path and os.path.exists(result_path):
                print(f"✓ Inbetween generated successfully in {format_time(frame_generation_time)}!")
                
                # Add to results
                inbetween_results.append({
                    'frame0': frame0_num,
                    'frame1': frame1_num,
                    'png_path': result_path,
                    'gif_path': gif_path,
                    'timestamp': timestamp,
                    'generation_time': frame_generation_time
                })
                
                # Add frames to the sequence for final GIF
                if i == 0:  # First iteration, add the first keyframe
                    all_frame_paths.append(img0_path)
                all_frame_paths.append(result_path)  # Add the inbetween
                all_frame_paths.append(img1_path)    # Add the second keyframe
                
            else:
                print(f"✗ Failed to generate inbetween for {frame0_num} → {frame1_num}")
                
        except Exception as e:
            frame_generation_time = time.time() - frame_start_time
            print(f"✗ Error generating inbetween for {frame0_num} → {frame1_num} after {format_time(frame_generation_time)}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create final combined GIF
    print(f"\nCreating final combined GIF for {experiment_name}...")
    gif_start_time = time.time()
    final_gif_path = os.path.join(experiment_output_dir, f"{experiment_name}_full_output.gif")
    create_combined_gif(all_frame_paths, final_gif_path, duration=0.1)
    gif_creation_time = time.time() - gif_start_time
    print(f"✓ Final GIF created in {format_time(gif_creation_time)}")
    
    # Calculate timing statistics
    if frame_generation_times:
        avg_generation_time = sum(frame_generation_times) / len(frame_generation_times)
        min_generation_time = min(frame_generation_times)
        max_generation_time = max(frame_generation_times)
    else:
        avg_generation_time = min_generation_time = max_generation_time = 0
    
    # Return experiment results
    return {
        'experiment_name': experiment_name,
        'inbetween_results': inbetween_results,
        'frame_generation_times': frame_generation_times,
        'wrapper_init_time': wrapper_init_time,
        'gif_creation_time': gif_creation_time,
        'avg_generation_time': avg_generation_time,
        'min_generation_time': min_generation_time,
        'max_generation_time': max_generation_time,
        'total_frames': len(inbetween_results),
        'output_dir': experiment_output_dir,
        'final_gif_path': final_gif_path
    }


def main():
    """Main function to generate inbetweens for sc367c frames with multiple models."""
    
    # Start overall timing
    overall_start_time = time.time()
    
    # Configuration
    cleaned_images_dir = "/Users/MakarovAleksandr/Downloads/cleaned_images"  # Absolute path to cleaned_images
    output_dir = "sc367c_experiment"
    temp_dir = "temp_sc367c"
    
    # Frame numbers to process (consecutive pairs)
    frame_numbers = ["0001", "0004", "0008", "0012", "0017", "0020", "0023", "0026", "0028", "0030", "0031"]
    
    # Common parameters for all experiments
    device = "cpu"
    size = 1440
    no_edge_sharpen = True
    
    print("=" * 80)
    print("MULTI-MODEL INBETWEEN GENERATION FOR ep01_sc367c_Cyberslav_CLEAN_")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Size: {size}")
    print(f"Edge sharpening: {not no_edge_sharpen}")
    print(f"Output directory: {output_dir}")
    print(f"Frame numbers: {frame_numbers}")
    print("=" * 80)
    
    # Create main output and temp directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get frame paths
    frame_paths = get_frame_paths(cleaned_images_dir, frame_numbers)
    
    if len(frame_paths) != len(frame_numbers):
        print(f"Error: Expected {len(frame_numbers)} frames, found {len(frame_paths)}")
        sys.exit(1)
    
    print(f"\nFound {len(frame_paths)} frames:")
    for i, path in enumerate(frame_paths):
        print(f"  {i+1:2d}. {os.path.basename(path)}")
    
    # Define experiments (reorganized: RIFE and SAIN first, TPS last)
    experiments = []
    
    # Add RIFE experiment first
    experiments.append({
        'name': 'RIFE',
        'wrapper_class': RIFEInterpolator,
        'params': {
            'device': device
        }
    })
    
    # Add SAIN experiments
    experiments.append({
        'name': 'SAIN_basic',
        'wrapper_class': SAINWrapper,
        'params': {
            'device': device,
            'size': size
        }
    })
    
    experiments.append({
        'name': 'SAIN_enhanced',
        'wrapper_class': SAINWrapperEnhanced,
        'params': {
            'device': device,
            'size': size
        }
    })
    
    # Add TPS experiments
    experiments.append({
        'name': 'TPS_no_vector_cleanup',
        'wrapper_class': TPSInbetweenWrapper,
        'params': {
            'device': device,
            'no_edge_sharpen': True,
            'vector_cleanup': False,
            'uniform_thin': False
        }
    })
    
    experiments.append({
        'name': 'TPS_with_vector_cleanup',
        'wrapper_class': TPSInbetweenWrapper,
        'params': {
            'device': device,
            'no_edge_sharpen': True,
            'vector_cleanup': True,
            'uniform_thin': False
        }
    })
    
    # Run all experiments
    all_results = []
    
    for experiment_config in experiments:
        experiment_name = experiment_config['name']
        wrapper_class = experiment_config['wrapper_class']
        params = experiment_config['params']
        
        print(f"\n{'='*80}")
        print(f"INITIALIZING EXPERIMENT: {experiment_name}")
        print(f"{'='*80}")
        print(f"Wrapper: {wrapper_class.__name__}")
        print(f"Parameters: {params}")
        
        # Initialize wrapper
        wrapper_init_start = time.time()
        wrapper = wrapper_class(**params)
        wrapper_init_time = time.time() - wrapper_init_start
        print(f"✓ Wrapper initialized in {format_time(wrapper_init_time)}")
        
        # Run experiment
        result = run_experiment(
            experiment_name=experiment_name,
            wrapper=wrapper,
            frame_paths=frame_paths,
            frame_numbers=frame_numbers,
            output_dir=output_dir,
            temp_dir=temp_dir,
            size=size
        )
        
        all_results.append(result)
    
    # Calculate overall timing
    overall_time = time.time() - overall_start_time
    
    # Final summary
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Total elapsed time: {format_time(overall_time)}")
    
    print(f"\nEXPERIMENT RESULTS:")
    print(f"{'='*80}")
    
    for result in all_results:
        print(f"\n{result['experiment_name']}:")
        print(f"  Total frames: {result['total_frames']}")
        print(f"  Output directory: {result['output_dir']}")
        print(f"  Final GIF: {os.path.basename(result['final_gif_path'])}")
        print(f"  Timing:")
        print(f"    Wrapper init: {format_time(result['wrapper_init_time'])}")
        print(f"    Avg per frame: {format_time(result['avg_generation_time'])}")
        print(f"    Min frame: {format_time(result['min_generation_time'])}")
        print(f"    Max frame: {format_time(result['max_generation_time'])}")
        print(f"    GIF creation: {format_time(result['gif_creation_time'])}")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    
    # Compare average generation times
    print(f"Average generation times per frame:")
    for result in all_results:
        print(f"  {result['experiment_name']:20s}: {format_time(result['avg_generation_time'])}")
    
    print(f"\nTotal experiment times (wrapper init + all frames + GIF):")
    for result in all_results:
        total_time = result['wrapper_init_time'] + (result['avg_generation_time'] * result['total_frames']) + result['gif_creation_time']
        print(f"  {result['experiment_name']:20s}: {format_time(total_time)}")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 