#!/usr/bin/env python3
"""
Clean experiment script for generating inbetweens on multiple scenes.

This script processes consecutive pairs of keyframes and generates inbetween frames
with the following models:
- TPS-Inbetween (with/without vector cleanup)
- RIFE (baseline)

All experiments use:
- device = cpu
- size = 1440
- no_edge_sharpen = true
- no uniform_thin = true

The script creates individual inbetweens, final combined GIFs, and detailed log files.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
import imageio
from PIL import Image
import numpy as np

# Add current directory to path for wrapper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import wrappers
from wrappers.tps_inbetween_wrapper import TPSInbetweenWrapper
from wrappers.rife_wrapper import RIFEInterpolator

print("✓ All wrappers imported successfully")


# Scene configuration
SCENE_CONFIG = {
    'sc238': [139, 188],
    'sc277': [1, 3, 12, 15, 20, 21, 24, 27, 30, 35, 38, 39, 42, 48, 56, 60, 68],
    'sc281': [1, 3, 5, 12, 13, 15, 17, 19, 28, 36],
    'sc288': [1, 13, 16, 17, 20, 28, 31, 38],
    'sc324': [1, 4, 5, 6, 8, 10, 12, 13, 16, 19, 20, 22, 26, 27],
    'sc326': [8, 10, 14, 19, 22, 23, 26],
    'sc367b': [1, 3, 6, 7],
    'sc367c': [1, 4, 8, 12, 17, 20, 23, 26, 28, 30, 31]
}


def write_log(log_file_path, message):
    """Write a message to the log file with timestamp."""
    with open(log_file_path, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")


def get_frame_paths(cleaned_images_dir, scene_name, frame_numbers):
    """
    Get the full paths for the specified frame numbers for a given scene.
    
    Args:
        cleaned_images_dir (str): Path to the cleaned_images directory
        scene_name (str): Scene name (e.g., 'sc367c')
        frame_numbers (list): List of frame numbers as integers
    
    Returns:
        list: List of full file paths
    """
    frame_paths = []
    for frame_num in frame_numbers:
        # Pad frame number to 4 digits
        padded_frame = str(frame_num).zfill(4)
        filename = f"ep01_{scene_name}_Cyberslav_CLEAN_{padded_frame}.jpg"
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


def run_experiment(experiment_name, wrapper, frame_paths, frame_numbers, output_dir, temp_dir, size, scene_name):
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
        scene_name (str): Scene name for logging
    
    Returns:
        dict: Experiment results with timing and file paths
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name} - {scene_name}")
    print(f"{'='*80}")
    
    # Create experiment-specific directories
    experiment_output_dir = os.path.join(output_dir, experiment_name)
    experiment_temp_dir = os.path.join(temp_dir, experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)
    os.makedirs(experiment_temp_dir, exist_ok=True)
    
    # Create log file
    log_file_path = os.path.join(experiment_output_dir, f"{scene_name}_{experiment_name}_log.txt")
    
    # Initialize log file
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Experiment Log: {experiment_name} - {scene_name}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Frame numbers: {frame_numbers}\n")
        f.write(f"Output directory: {experiment_output_dir}\n")
        f.write("="*80 + "\n\n")
    
    # Initialize wrapper timing
    wrapper_init_start = time.time()
    print(f"Initializing {experiment_name} wrapper...")
    wrapper_init_time = time.time() - wrapper_init_start
    print(f"✓ Wrapper ready in {format_time(wrapper_init_time)}")
    
    write_log(log_file_path, f"Wrapper initialization time: {format_time(wrapper_init_time)}")
    
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
        print(f"Processing pair {i+1}/{len(frame_paths)-1}: {frame0_num:04d} → {frame1_num:04d}")
        print(f"{'='*60}")
        
        # Generate timestamp for this inbetween
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output filename
        output_filename = f"{timestamp}_{frame0_num:04d}_to_{frame1_num:04d}"
        output_path = os.path.join(experiment_output_dir, f"{output_filename}.png")
        gif_path = os.path.join(experiment_output_dir, f"{output_filename}.gif")
        
        print(f"Input 0: {os.path.basename(img0_path)}")
        print(f"Input 1: {os.path.basename(img1_path)}")
        print(f"Output PNG: {output_filename}.png")
        print(f"Output GIF: {output_filename}.gif")
        
        # Start timing for this frame pair
        frame_start_time = time.time()
        
        write_log(log_file_path, f"Starting frame pair {frame0_num:04d} → {frame1_num:04d}")
        
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
            
        else:
            raise ValueError(f"Unknown wrapper type: {type(wrapper)}")
        
        # Calculate generation time for this frame
        frame_generation_time = time.time() - frame_start_time
        frame_generation_times.append(frame_generation_time)
        
        if result_path and os.path.exists(result_path):
            print(f"✓ Inbetween generated successfully in {format_time(frame_generation_time)}!")
            write_log(log_file_path, f"Frame pair {frame0_num:04d} → {frame1_num:04d} completed in {format_time(frame_generation_time)}")
            
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
            raise RuntimeError(f"Failed to generate inbetween for {frame0_num:04d} → {frame1_num:04d} - no output file created")
    
    # Create final combined GIF
    print(f"\nCreating final combined GIF for {experiment_name}...")
    gif_start_time = time.time()
    final_gif_path = os.path.join(experiment_output_dir, f"{scene_name}_{experiment_name}_full_output.gif")
    create_combined_gif(all_frame_paths, final_gif_path, duration=0.1)
    gif_creation_time = time.time() - gif_start_time
    print(f"✓ Final GIF created in {format_time(gif_creation_time)}")
    
    write_log(log_file_path, f"Final GIF creation time: {format_time(gif_creation_time)}")
    
    # Calculate timing statistics
    if frame_generation_times:
        avg_generation_time = sum(frame_generation_times) / len(frame_generation_times)
        min_generation_time = min(frame_generation_times)
        max_generation_time = max(frame_generation_times)
        total_generation_time = sum(frame_generation_times)
    else:
        avg_generation_time = min_generation_time = max_generation_time = total_generation_time = 0
    
    # Write final statistics to log
    write_log(log_file_path, "\n" + "="*50)
    write_log(log_file_path, "FINAL STATISTICS")
    write_log(log_file_path, "="*50)
    write_log(log_file_path, f"Total frames processed: {len(frame_generation_times)}")
    write_log(log_file_path, f"Average generation time per frame: {format_time(avg_generation_time)}")
    write_log(log_file_path, f"Minimum generation time: {format_time(min_generation_time)}")
    write_log(log_file_path, f"Maximum generation time: {format_time(max_generation_time)}")
    write_log(log_file_path, f"Total generation time: {format_time(total_generation_time)}")
    write_log(log_file_path, f"Wrapper initialization time: {format_time(wrapper_init_time)}")
    write_log(log_file_path, f"GIF creation time: {format_time(gif_creation_time)}")
    write_log(log_file_path, f"Total experiment time: {format_time(wrapper_init_time + total_generation_time + gif_creation_time)}")
    
    # Write individual frame times
    write_log(log_file_path, "\nINDIVIDUAL FRAME TIMES:")
    for i, time_val in enumerate(frame_generation_times):
        frame0_num = frame_numbers[i]
        frame1_num = frame_numbers[i + 1]
        write_log(log_file_path, f"Frame {frame0_num:04d} → {frame1_num:04d}: {format_time(time_val)}")
    
    write_log(log_file_path, f"\nExperiment completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return experiment results
    return {
        'experiment_name': experiment_name,
        'scene_name': scene_name,
        'inbetween_results': inbetween_results,
        'frame_generation_times': frame_generation_times,
        'wrapper_init_time': wrapper_init_time,
        'gif_creation_time': gif_creation_time,
        'avg_generation_time': avg_generation_time,
        'min_generation_time': min_generation_time,
        'max_generation_time': max_generation_time,
        'total_generation_time': total_generation_time,
        'total_frames': len(inbetween_results),
        'output_dir': experiment_output_dir,
        'final_gif_path': final_gif_path,
        'log_file_path': log_file_path
    }


def main():
    """Main function to generate inbetweens for multiple scenes with TPS and RIFE models."""
    
    # Start overall timing
    overall_start_time = time.time()
    
    # Configuration
    cleaned_images_dir = "/Users/MakarovAleksandr/Downloads/cleaned_images"  # Absolute path to cleaned_images
    output_dir = "experiments"
    temp_dir = "temp_experiments"
    
    # Common parameters for all experiments
    device = "cpu"
    size = 1440
    no_edge_sharpen = True
    
    print("=" * 80)
    print("MULTI-SCENE MULTI-MODEL INBETWEEN GENERATION")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Size: {size}")
    print(f"Edge sharpening: {not no_edge_sharpen}")
    print(f"Output directory: {output_dir}")
    print(f"Scenes: {list(SCENE_CONFIG.keys())}")
    print("=" * 80)
    
    # Create main output and temp directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Define experiments (only TPS and RIFE)
    experiments = [
        {
            'name': 'RIFE',
            'wrapper_class': RIFEInterpolator,
            'params': {
                'device': device
            }
        },
        {
            'name': 'TPS_no_vector_cleanup',
            'wrapper_class': TPSInbetweenWrapper,
            'params': {
                'device': device,
                'no_edge_sharpen': True,
                'vector_cleanup': False,
                'uniform_thin': False
            }
        },
        {
            'name': 'TPS_with_vector_cleanup',
            'wrapper_class': TPSInbetweenWrapper,
            'params': {
                'device': device,
                'no_edge_sharpen': True,
                'vector_cleanup': True,
                'uniform_thin': False
            }
        }
    ]
    
    # Run experiments for each scene
    all_results = []
    
    for scene_name, frame_numbers in SCENE_CONFIG.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING SCENE: {scene_name}")
        print(f"{'='*80}")
        print(f"Frame numbers: {frame_numbers}")
        
        # Create scene-specific directories
        scene_output_dir = os.path.join(output_dir, scene_name)
        scene_temp_dir = os.path.join(temp_dir, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)
        os.makedirs(scene_temp_dir, exist_ok=True)
        
        # Get frame paths for this scene
        frame_paths = get_frame_paths(cleaned_images_dir, scene_name, frame_numbers)
        
        if len(frame_paths) != len(frame_numbers):
            print(f"Warning: Expected {len(frame_numbers)} frames, found {len(frame_paths)} for {scene_name}")
            if len(frame_paths) < 2:
                print(f"Skipping {scene_name} - not enough frames found")
                continue
        
        print(f"\nFound {len(frame_paths)} frames for {scene_name}:")
        for i, path in enumerate(frame_paths):
            print(f"  {i+1:2d}. {os.path.basename(path)}")
        
        # Run all experiments for this scene
        for experiment_config in experiments:
            experiment_name = experiment_config['name']
            wrapper_class = experiment_config['wrapper_class']
            params = experiment_config['params']
            
            print(f"\n{'='*80}")
            print(f"INITIALIZING EXPERIMENT: {experiment_name} - {scene_name}")
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
                output_dir=scene_output_dir,
                temp_dir=scene_temp_dir,
                size=size,
                scene_name=scene_name
            )
            
            all_results.append(result)
    
    # Calculate overall timing
    overall_time = time.time() - overall_start_time
    
    # Create overall summary log
    summary_log_path = os.path.join(output_dir, "overall_summary.txt")
    with open(summary_log_path, 'w', encoding='utf-8') as f:
        f.write("MULTI-SCENE MULTI-MODEL EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total scenes: {len(SCENE_CONFIG)}\n")
        f.write(f"Total experiments: {len(all_results)}\n")
        f.write(f"Total elapsed time: {format_time(overall_time)}\n\n")
        
        f.write("SCENE CONFIGURATION:\n")
        for scene_name, frame_numbers in SCENE_CONFIG.items():
            f.write(f"  {scene_name}: {frame_numbers}\n")
        f.write("\n")
        
        f.write("EXPERIMENT RESULTS:\n")
        f.write("=" * 80 + "\n")
        
        for result in all_results:
            f.write(f"\n{result['scene_name']} - {result['experiment_name']}:\n")
            f.write(f"  Total frames: {result['total_frames']}\n")
            f.write(f"  Output directory: {result['output_dir']}\n")
            f.write(f"  Log file: {result['log_file_path']}\n")
            f.write(f"  Timing:\n")
            f.write(f"    Wrapper init: {format_time(result['wrapper_init_time'])}\n")
            f.write(f"    Avg per frame: {format_time(result['avg_generation_time'])}\n")
            f.write(f"    Min frame: {format_time(result['min_generation_time'])}\n")
            f.write(f"    Max frame: {format_time(result['max_generation_time'])}\n")
            f.write(f"    Total generation: {format_time(result['total_generation_time'])}\n")
            f.write(f"    GIF creation: {format_time(result['gif_creation_time'])}\n")
    
    # Final summary
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total scenes: {len(SCENE_CONFIG)}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Total elapsed time: {format_time(overall_time)}")
    print(f"Summary log saved: {summary_log_path}")
    
    print(f"\nEXPERIMENT RESULTS BY SCENE:")
    print(f"{'='*80}")
    
    for scene_name in SCENE_CONFIG.keys():
        scene_results = [r for r in all_results if r['scene_name'] == scene_name]
        if scene_results:
            print(f"\n{scene_name}:")
            for result in scene_results:
                print(f"  {result['experiment_name']:25s}: {result['total_frames']} frames, avg {format_time(result['avg_generation_time'])}")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    
    # Compare average generation times by model
    print(f"Average generation times per frame by model:")
    for experiment_name in ['RIFE', 'TPS_no_vector_cleanup', 'TPS_with_vector_cleanup']:
        model_results = [r for r in all_results if r['experiment_name'] == experiment_name]
        if model_results:
            avg_time = sum(r['avg_generation_time'] for r in model_results) / len(model_results)
            print(f"  {experiment_name:25s}: {format_time(avg_time)} (across {len(model_results)} scenes)")
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 