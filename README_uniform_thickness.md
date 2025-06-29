# Uniform-Thickness Skeletonization for TPS-Inbetween

## Overview

The uniform-thickness skeletonization feature ensures consistent stroke thickness throughout the TPS-Inbetween interpolation process. This addresses the issue where unequal stroke widths in key frames can cause aliasing and make the network invent its own thickness in the inbetween frames, resulting in uneven lines.

## How It Works

1. **Pre-processing**: Input key frames are thinned to 1-pixel skeletons using morphological skeletonization
2. **Inference**: The TPS model processes these uniform-thickness inputs
3. **Post-processing**: Both input keys and generated frames are re-dilated to restore the desired stroke thickness

This approach forces a single, consistent stroke width through the entire triplet (key1 → inbetween → key2).

## Installation

Install the required dependency:

```bash
pip install scikit-image
```

## Usage

### Command Line Interface

Enable uniform-thickness processing with the `--uniform_thin` flag:

```bash
python wrappers/tps_inbetween_wrapper.py \
    --img0 input1.png \
    --img1 input2.png \
    --uniform_thin \
    --output result.png
```

### Python API

```python
from wrappers.tps_inbetween_wrapper import TPSInbetweenWrapper

# Initialize wrapper with uniform thickness enabled
wrapper = TPSInbetweenWrapper(uniform_thin=True)

# Generate inbetween frame
output_path = wrapper.interpolate(
    img0_path="input1.png",
    img1_path="input2.png",
    output_path="result.png"
)
```

### Example Script

Use the dedicated example script:

```bash
python example_usage_tps_uniform_thin.py \
    --img0 input1.png \
    --img1 input2.png \
    --uniform_thin \
    --thickness 2
```

## Parameters

- `--uniform_thin`: Enable uniform-thickness skeletonization
- `--thickness`: Stroke thickness for rehydration (default: 2 pixels)

## Technical Details

### Skeletonization Process

1. **Thinning**: Uses `skimage.morphology.skeletonize()` to reduce strokes to 1-pixel width
2. **Binary Conversion**: Converts grayscale images to binary using threshold < 128
3. **Skeleton Extraction**: Preserves connectivity while minimizing stroke width

### Rehydration Process

1. **Binary Conversion**: Converts output back to binary format
2. **Dilation**: Uses `skimage.morphology.dilation()` with square kernel
3. **Thickness Control**: Kernel size determines final stroke thickness

### Implementation Details

- **Pre-processing**: Applied in `_generate_matches()` method after loading input images
- **Post-processing**: Applied in all output generation methods (`interpolate`, `interpolate_sequence`, `_create_simple_gif`)
- **Compatibility**: Works with all existing features (edge sharpening, vector cleanup, etc.)

## Benefits

1. **Consistent Thickness**: Eliminates stroke width variations in inbetween frames
2. **Reduced Aliasing**: Prevents the network from inventing inconsistent line thicknesses
3. **Better Quality**: Produces more professional-looking line art interpolation
4. **Configurable**: Adjustable output thickness to match your artistic style

## Limitations

- **Processing Time**: Additional morphological operations add slight overhead
- **Detail Loss**: Very fine details may be simplified during skeletonization
- **Thickness Control**: Output thickness is uniform across all strokes

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure `scikit-image` is installed
2. **Memory Issues**: Large images may require reducing `max_image_size`
3. **Quality Loss**: Try adjusting the `thickness` parameter for better results

### Debug Output

The wrapper provides verbose output when uniform thickness is enabled:

```
Applying skeletonization to input images for uniform thickness...
Uniform thickness: enabled
```

## Examples

### Before (Standard Processing)
- Uneven stroke thickness in inbetween frames
- Network-generated thickness variations
- Potential aliasing artifacts

### After (Uniform Thickness)
- Consistent stroke width throughout sequence
- Professional line art quality
- Smooth interpolation between key frames

## Integration

This feature integrates seamlessly with the existing Animation AI backend:

- Compatible with all TPS-Inbetween wrapper methods
- Works alongside edge sharpening and vector cleanup
- Maintains the same API interface
- Adds minimal overhead to processing pipeline 