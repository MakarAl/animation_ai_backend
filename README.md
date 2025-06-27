# Animation AI Backend

A consolidated backend for multiple state-of-the-art animation frame interpolation models, providing a unified interface for generating inbetween frames from two input images.

## 🎬 Supported Models

This backend includes four powerful frame interpolation models:

| Model | Description | Paper | Best For |
|-------|-------------|-------|----------|
| **TPS Inbetween** | Thin-Plate Spline based interpolation with feature matching | [Paper](https://arxiv.org/abs/2203.16571) | General purpose, high quality |
| **RIFE** | Real-time Intermediate Flow Estimation | [Paper](https://arxiv.org/abs/2011.06294) | Real-time applications |
| **SAIN** | Sketch-Aware Interpolation Network | [Paper](https://arxiv.org/abs/2203.16571) | Line art and sketches |
| **SAIN Enhanced** | Enhanced SAIN with stroke optimization | - | Improved line art quality |

## 📁 Project Structure

```
Animation_AI_backend/
├── models/                    # Model weights and implementations
│   ├── TPS_Inbetween/        # TPS Inbetween model files
│   ├── RIFE/                 # RIFE model files
│   ├── SAIN/                 # SAIN model files
│   └── AnimeInbet/           # AnimeInbet model files (removed)
├── wrappers/                 # Model wrapper implementations
│   ├── tps_inbetween_wrapper.py
│   ├── rife_wrapper.py
│   ├── sain_wrapper.py
│   └── sain_enhanced_wrapper.py
├── example_usage_*.py        # CLI scripts for each model
├── input_images/             # Sample input images
├── test_outputs/             # Generated outputs
├── utils/                    # Utility functions
└── requirements.txt          # Python dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Apple Silicon Mac (M1/M2) support via MPS

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Animation_AI_backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights** (if not already included):
   - TPS Inbetween: Weights should be in `models/TPS_Inbetween/ckpt/`
   - RIFE: Weights should be in `models/RIFE/`
   - SAIN: Weights should be in `models/SAIN/ckp/checkpoints/`

## 🎯 Quick Start

### Basic Usage

All models follow the same interface. Here's a quick example:

```bash
# Generate a single inbetween frame
python example_usage_tps_inbetween.py \
    --img0 input_images/frame1.jpg \
    --img1 input_images/frame2.jpg \
    --device cpu \
    --size 720

# Generate with GIF output
python example_usage_tps_inbetween.py \
    --img0 input_images/frame1.jpg \
    --img1 input_images/frame2.jpg \
    --device cuda \
    --size 1024 \
    --gif
```

### Output Files

When using the `--gif` flag, both files are generated:
- `<timestamp>_<model_name>_test.png` - Single interpolated frame
- `<timestamp>_<model_name>_test.gif` - Animation sequence (frame1 → inbetween → frame2)

## 📖 Detailed Usage

### TPS Inbetween

**Best for:** General purpose interpolation with high quality results

```bash
python example_usage_tps_inbetween.py \
    --img0 input_images/frame1.jpg \
    --img1 input_images/frame2.jpg \
    --device cuda \
    --size 720 \
    --gif \
    --num_frames 1 \
    --gif_duration 0.05
```

**Parameters:**
- `--img0`, `--img1`: Input image paths
- `--size`: Maximum image dimension (default: 720)
- `--device`: Device to use (cpu/cuda/mps, auto-detect if not specified)
- `--gif`: Create GIF output
- `--num_frames`: Number of intermediate frames (default: 1)
- `--gif_duration`: Duration per frame in seconds (default: 0.05)

### RIFE

**Best for:** Real-time applications and video processing

```bash
python example_usage_rife.py \
    --img0 input_images/frame1.jpg \
    --img1 input_images/frame2.jpg \
    --device cuda \
    --size 720 \
    --gif \
    --num_frames 1
```

**Parameters:**
- `--img0`, `--img1`: Input image paths
- `--size`: Maximum image dimension (default: 720)
- `--device`: Device to use (cpu/cuda/mps, auto-detect if not specified)
- `--gif`: Create GIF output
- `--num_frames`: Number of intermediate frames (default: 1)

### SAIN

**Best for:** Line art, sketches, and cartoon-style animations

```bash
python example_usage_sain.py \
    input_images/frame1.jpg \
    input_images/frame2.jpg \
    --device cuda \
    --size 512 \
    --gif
```

**Parameters:**
- `img0`, `img1`: Input image paths (positional arguments)
- `--size`: Input image size, must be divisible by 8 (default: 512)
- `--device`: Device to use (cpu/cuda/mps, auto-detect if not specified)
- `--gif`: Create GIF output

### SAIN Enhanced

**Best for:** Improved line art quality with stroke optimization

```bash
python example_usage_sain_enhanced.py \
    input_images/frame1.jpg \
    input_images/frame2.jpg \
    --device cuda \
    --size 1024 \
    --gif \
    --dilation-radius 2 \
    --noise-amplitude 0.05 \
    --debug-flow
```

**Parameters:**
- `img0`, `img1`: Input image paths (positional arguments)
- `--size`: Input image size, must be divisible by 8 (default: 1024)
- `--device`: Device to use (cpu/cuda/mps, auto-detect if not specified)
- `--gif`: Create GIF output
- `--no-enhance`: Disable stroke enhancement
- `--dilation-radius`: Stroke dilation radius (default: 2)
- `--noise-amplitude`: Synthetic noise amplitude (default: 0.05)
- `--debug-flow`: Save flow visualizations for debugging

## 🔧 Advanced Usage

### Programmatic Interface

You can also use the wrappers directly in your Python code:

```python
from wrappers.tps_inbetween_wrapper import TPSInbetweenWrapper
from wrappers.rife_wrapper import RIFEInterpolator
from wrappers.sain_wrapper import sain_interpolate
from wrappers.sain_enhanced_wrapper import sain_enhanced_interpolate

# TPS Inbetween
wrapper = TPSInbetweenWrapper(device='cuda')
result_path = wrapper.interpolate(
    img0_path='frame1.jpg',
    img1_path='frame2.jpg',
    output_path='output.png',
    num_frames=1,
    max_image_size=720
)

# RIFE
wrapper = RIFEInterpolator(device='cuda')
result_imgs = wrapper.interpolate(
    img0='frame1.jpg',
    img1='frame2.jpg',
    exp=1,
    max_size=720
)

# SAIN
result = sain_interpolate(
    img0_path='frame1.jpg',
    img1_path='frame2.jpg',
    t=0.5,
    device='cuda',
    size=512
)

# SAIN Enhanced
result = sain_enhanced_interpolate(
    img0_path='frame1.jpg',
    img1_path='frame2.jpg',
    t=0.5,
    device='cuda',
    size=1024,
    enhance_strokes=True
)
```

### Batch Processing

For processing multiple image pairs:

```bash
# Process multiple pairs with TPS Inbetween
for pair in pairs/*; do
    python example_usage_tps_inbetween.py \
        --img0 "$pair/frame1.jpg" \
        --img1 "$pair/frame2.jpg" \
        --device cuda \
        --size 720 \
        --gif
done
```

## 🎨 Output Quality

### GIF Quality Settings

All models use high-quality GIF settings:
- `optimize=False`: Disables aggressive optimization
- `quality=10`: Uses higher quality (lower compression)

### File Size Comparison

Typical file sizes for 720p output:
- **PNG**: 15-50KB (lossless)
- **High-quality GIF**: 25-60KB (good quality)
- **Standard GIF**: 15-30KB (compressed)

## 🖥️ Performance

### Device Recommendations

| Device | Speed | Memory Usage | Best For |
|--------|-------|--------------|----------|
| **CUDA GPU** | Fastest | High | Production, batch processing |
| **MPS (Apple Silicon)** | Fast | Medium | Mac development |
| **CPU** | Slowest | Low | Testing, small images |

### Memory Requirements

| Model | 720p | 1080p | 4K |
|-------|------|-------|----|
| TPS Inbetween | 2-4GB | 4-8GB | 8-16GB |
| RIFE | 1-2GB | 2-4GB | 4-8GB |
| SAIN | 2-3GB | 3-6GB | 6-12GB |
| SAIN Enhanced | 2-4GB | 4-8GB | 8-16GB |

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce `--size` parameter
   - Use CPU device instead
   - Close other GPU applications

2. **Model not found:**
   - Ensure model weights are in correct directories
   - Check file permissions

3. **Poor GIF quality:**
   - GIFs are automatically saved with high quality settings
   - Use PNG output for maximum quality

4. **Slow processing:**
   - Use CUDA GPU if available
   - Reduce image size
   - Use RIFE for real-time applications

### Debug Mode

Enable debug output for SAIN Enhanced:
```bash
python example_usage_sain_enhanced.py \
    frame1.jpg frame2.jpg \
    --debug-flow
```

This saves optical flow visualizations to `debug_flow/` directory.

## 📝 Examples

### Example 1: Basic Interpolation
```bash
# Generate a single inbetween frame
python example_usage_tps_inbetween.py \
    --img0 input_images/ep01_sc324_Cyberslav_CLEAN_0007.jpg \
    --img1 input_images/ep01_sc324_Cyberslav_CLEAN_0008.jpg \
    --device cpu \
    --size 512
```

### Example 2: High-Quality Animation
```bash
# Generate high-quality GIF with multiple frames
python example_usage_rife.py \
    --img0 input_images/frame1.jpg \
    --img1 input_images/frame2.jpg \
    --device cuda \
    --size 1024 \
    --gif \
    --num_frames 3
```

### Example 3: Line Art Processing
```bash
# Process line art with enhanced SAIN
python example_usage_sain_enhanced.py \
    input_images/sketch1.jpg \
    input_images/sketch2.jpg \
    --device cuda \
    --size 1024 \
    --gif \
    --dilation-radius 3 \
    --noise-amplitude 0.1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with multiple models
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TPS Inbetween**: Original authors for the TPS-based interpolation
- **RIFE**: Real-time Intermediate Flow Estimation team
- **SAIN**: Sketch-Aware Interpolation Network researchers
- **AnimeInbet**: Deep Geometrized Cartoon Line Inbetweening team

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information

---

**Happy animating! 🎬✨** 