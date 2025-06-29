# Animation AI - Multi-Model Experiment Framework

## Overview
This framework provides comprehensive testing and comparison of different inbetween generation models on the same dataset, with detailed performance metrics and output organization.

## Quick Start

### Basic Usage
```bash
python generate_sc367c_inbetweens.py
```

### Configuration
The script automatically configures:
- **Resolution**: 1440p (2560x1440)
- **Device**: CPU (configurable)
- **Input**: ep01_sc367c_Cyberslav_CLEAN_ sequence
- **Output**: Organized by model in `sc367c_experiment/` directory

## Supported Models

### 1. RIFE (Real-time Intermediate Flow Estimation)
- **Speed**: ~2.5 seconds per frame
- **Quality**: High-quality frame interpolation
- **Best for**: Real-time applications, smooth motion

### 2. SAIN Basic
- **Speed**: ~4-5 minutes per frame
- **Quality**: Advanced inbetween generation
- **Best for**: High-quality animation sequences

### 3. SAIN Enhanced
- **Speed**: ~5-6 minutes per frame
- **Quality**: Enhanced version with improved results
- **Best for**: Premium quality output

### 4. TPS (Thin Plate Spline) - No Vector Cleanup
- **Speed**: Variable (typically slower)
- **Quality**: Vector-aware interpolation
- **Best for**: Line art and vector graphics

### 5. TPS (Thin Plate Spline) - With Vector Cleanup
- **Speed**: Variable (typically slower)
- **Quality**: Vector-aware interpolation with cleanup
- **Best for**: Clean line art with artifact removal

## Output Structure

```
sc367c_experiment/
├── RIFE/
│   ├── 20250629_224857_0001_to_0004.png
│   ├── 20250629_224857_0001_to_0004.gif
│   ├── ...
│   └── RIFE_full_output.gif
├── SAIN_basic/
│   ├── 20250629_224857_0001_to_0004.png
│   ├── 20250629_224857_0001_to_0004.gif
│   ├── ...
│   └── SAIN_basic_full_output.gif
├── SAIN_enhanced/
│   └── ...
├── TPS_no_vector_cleanup/
│   └── ...
└── TPS_with_vector_cleanup/
    └── ...
```

## Performance Metrics

The script provides detailed timing information:
- **Wrapper initialization time**
- **Per-frame processing time**
- **Total experiment time**
- **GIF creation time**
- **Min/Max frame times**

## Configuration Options

### Device Selection
```python
device = 'cpu'  # or 'cuda' for GPU
```

### Resolution Control
```python
size = 1440  # Output resolution (width)
```

### Input Frame Selection
```python
frame_numbers = ["0001", "0004", "0008", "0012", "0017", 
                "0020", "0023", "0026", "0028", "0030", "0031"]
```

## Recent Results

### Latest Benchmark (CPU, 1440p):
| Model | Avg Time/Frame | Total Time | Status |
|-------|----------------|------------|---------|
| RIFE | 2.50s | 34.51s | ✅ Working |
| SAIN_basic | 4m 43.99s | 47m 34.30s | ✅ Working |
| SAIN_enhanced | 5m 13.99s | 52m 42.10s | ✅ Working |
| TPS_no_vector_cleanup | TBD | TBD | 🔧 Fixed |
| TPS_with_vector_cleanup | TBD | TBD | 🔧 Fixed |

## Error Handling

The script includes comprehensive error handling:
- **Import errors**: Graceful fallback and detailed reporting
- **Model loading errors**: Clear error messages with troubleshooting
- **Processing errors**: Individual frame failure tracking
- **Memory errors**: Automatic resolution scaling

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all model directories are in the correct location
   - Check that sys.path includes necessary directories

2. **Memory Issues**
   - Reduce resolution by changing `size` parameter
   - Use CPU instead of GPU if memory is limited

3. **Model Loading Failures**
   - Verify model checkpoint files exist
   - Check device compatibility (CPU/GPU)

### Debug Mode
Add debug prints by modifying the script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Customization

### Adding New Models
1. Create wrapper class following existing pattern
2. Add to experiments list
3. Update documentation

### Modifying Input Data
1. Change `cleaned_images_dir` path
2. Update `frame_numbers` list
3. Adjust file naming patterns

### Output Customization
1. Modify output directory structure
2. Change file naming conventions
3. Add custom metrics collection

## Dependencies

### Required Packages
- torch
- torchvision
- opencv-python
- numpy
- pillow
- scikit-image
- matplotlib

### Model-Specific Dependencies
- RIFE: Custom model files
- SAIN: Custom model files
- TPS: Custom model files + GlueStick

## Future Enhancements

### #AI-TODO: Planned Features
- GPU acceleration for all models
- Batch processing capabilities
- Real-time progress monitoring
- Quality assessment metrics
- Automated model selection
- Web interface integration

### #AI-TODO: Performance Improvements
- Memory optimization
- Parallel processing
- Progressive resolution
- Caching mechanisms

## Contributing

When adding new models or features:
1. Follow existing code patterns
2. Add comprehensive error handling
3. Update documentation
4. Include performance benchmarks
5. Add to experiment framework

## License

This experiment framework is part of the Animation AI project. 