import sys
import os

# Add the wrappers directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wrappers'))

def test_tps_import():
    """Test if we can import the TPS wrapper."""
    try:
        from tps_inbetween_wrapper import TPSInbetweenWrapper
        print("✅ TPS wrapper import successful")
        return True
    except Exception as e:
        print(f"❌ TPS wrapper import failed: {e}")
        return False

def test_tps_initialization():
    """Test if we can initialize the TPS wrapper."""
    try:
        from tps_inbetween_wrapper import TPSInbetweenWrapper
        
        print("Initializing TPS wrapper...")
        wrapper = TPSInbetweenWrapper(
            device='cpu',
            no_edge_sharpen=True,
            vector_cleanup=False,
            uniform_thin=False
        )
        print("✅ TPS wrapper initialization successful")
        return wrapper
    except Exception as e:
        print(f"❌ TPS wrapper initialization failed: {e}")
        return None

def test_tps_interpolation(wrapper):
    """Test if we can perform interpolation (with sample images if available)."""
    try:
        # Check if we have sample images
        sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input_images')
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(sample_files) >= 2:
                img0_path = os.path.join(sample_dir, sample_files[0])
                img1_path = os.path.join(sample_dir, sample_files[1])
                
                print(f"Testing interpolation with: {img0_path} and {img1_path}")
                
                # Create temp directory
                os.makedirs("temp", exist_ok=True)
                
                # Perform interpolation
                result_path = wrapper.interpolate(
                    img0_path=img0_path,
                    img1_path=img1_path,
                    output_path="temp/test_interpolation.png",
                    num_frames=1,
                    max_image_size=512,
                    create_gif=False
                )
                
                if result_path and os.path.exists(result_path):
                    print(f"✅ TPS interpolation successful: {result_path}")
                    return True
                else:
                    print("❌ TPS interpolation failed - no result file")
                    return False
            else:
                print("⚠️  No sample images found for testing")
                return None
        else:
            print("⚠️  No input_images directory found for testing")
            return None
            
    except Exception as e:
        print(f"❌ TPS interpolation test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing TPS Integration...")
    print("=" * 50)
    
    # Test 1: Import
    if not test_tps_import():
        sys.exit(1)
    
    # Test 2: Initialization
    wrapper = test_tps_initialization()
    if wrapper is None:
        sys.exit(1)
    
    # Test 3: Interpolation (if sample images available)
    result = test_tps_interpolation(wrapper)
    if result is False:
        sys.exit(1)
    
    print("=" * 50)
    print("✅ TPS model integration test completed successfully!")
    print("The TPS wrapper is ready to be used in the backend.")
