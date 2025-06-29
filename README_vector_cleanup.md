# Vector-Guidance Clean-up (Potrace) Feature

## Overview

The vector-guidance clean-up feature uses Potrace to convert raster images to vector paths and back to raster, which helps smooth micro-jitter, equalize stroke thickness, and provide cleaner line art. This acts like automated "ink & paint cleanup" that you would traditionally do by hand.

## What it does

- **Smooths micro-jitter**: Eliminates 1-2 pixel flickering between frames
- **Equalizes stroke thickness**: Makes line weights more consistent
- **Cleans up artifacts**: Removes noise and improves line quality
- **Reversible**: Can be enabled/disabled without affecting the core model

## Installation

Install the required dependencies:

```bash
pip install py-potrace cairosvg svgwrite
```

Or install from the requirements file:

```bash
pip install -r requirements_vector_cleanup.txt
```

## Usage

### Python API

```python
from wrappers.tps_inbetween_wrapper import TPSInbetweenWrapper

# Initialize with vector cleanup enabled
wrapper = TPSInbetweenWrapper(
    vector_cleanup=True,  # Enable vector cleanup
    use_cpu=True
)

# Generate inbetween frame with cleanup
output_path = wrapper.interpolate(
    "input1.jpg", 
    "input2.jpg",
    num_frames=1
)
```

### Command Line

```bash
# Basic usage with vector cleanup
python wrappers/tps_inbetween_wrapper.py \
    --img0 input1.jpg \
    --img1 input2.jpg \
    --vector_cleanup

# With additional options
python wrappers/tps_inbetween_wrapper.py \
    --img0 input1.jpg \
    --img1 input2.jpg \
    --vector_cleanup \
    --num_frames 3 \
    --create_gif \
    --cpu
```

## How it works

1. **Raster to Vector**: Converts the raster image to black and white, then traces it with Potrace
2. **Vector Processing**: Creates SVG paths from the traced curves
3. **Vector to Raster**: Renders the SVG back to raster using CairoSVG
4. **Stroke Control**: Uses configurable stroke width (default: 2px)

## Parameters

- `vector_cleanup` (bool): Enable/disable the feature
- `stroke` (int): Stroke width in pixels (default: 2)

## Testing

Run the test script to verify the functionality:

```bash
cd Animation_AI_backend
python test_vector_cleanup.py
```

This will generate comparison images in the `test_outputs/` directory.

## Performance Notes

- Vector cleanup adds processing time but improves visual quality
- Works best with line art and cartoon-style images
- May not be beneficial for photographic or highly detailed images
- CPU-only operation (doesn't use GPU acceleration)

## Integration Points

The vector cleanup is applied at these points in the pipeline:

1. Main output frame generation
2. Intermediate frame saving
3. GIF frame creation
4. Sequence generation

All integration points are conditional and only activate when `vector_cleanup=True`. 