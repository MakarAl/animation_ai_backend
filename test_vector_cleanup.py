#!/usr/bin/env python3
"""
Test script for the TPS-Inbetween wrapper with vector cleanup functionality.
"""

import os
import sys
from wrappers.tps_inbetween_wrapper import TPSInbetweenWrapper

def test_vector_cleanup():
    """Test the vector cleanup functionality."""
    
    # Check if we have test images
    test_img0 = "cleaned_images/ep01_sc277_Cyberslav_CLEAN_0001.jpg"
    test_img1 = "cleaned_images/ep01_sc277_Cyberslav_CLEAN_0002.jpg"
    
    if not os.path.exists(test_img0) or not os.path.exists(test_img1):
        print("Test images not found. Please ensure cleaned_images directory contains test images.")
        return False
    
    print("Testing TPS-Inbetween wrapper with vector cleanup...")
    
    try:
        # Test without vector cleanup
        print("\n1. Testing without vector cleanup...")
        wrapper_no_cleanup = TPSInbetweenWrapper(
            use_cpu=True,
            vector_cleanup=False
        )
        
        output_no_cleanup = wrapper_no_cleanup.interpolate(
            test_img0, test_img1,
            output_path="./test_outputs/no_cleanup.png",
            num_frames=1,
            temp_dir="./test_outputs"
        )
        
        print(f"Output without cleanup: {output_no_cleanup}")
        
        # Test with vector cleanup
        print("\n2. Testing with vector cleanup...")
        wrapper_with_cleanup = TPSInbetweenWrapper(
            use_cpu=True,
            vector_cleanup=True
        )
        
        output_with_cleanup = wrapper_with_cleanup.interpolate(
            test_img0, test_img1,
            output_path="./test_outputs/with_cleanup.png",
            num_frames=1,
            temp_dir="./test_outputs"
        )
        
        print(f"Output with cleanup: {output_with_cleanup}")
        
        print("\n✅ Vector cleanup test completed successfully!")
        print("Check the test_outputs directory for comparison images.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    # Create test output directory
    os.makedirs("./test_outputs", exist_ok=True)
    
    success = test_vector_cleanup()
    sys.exit(0 if success else 1) 