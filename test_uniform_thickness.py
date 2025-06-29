#!/usr/bin/env python3
"""
Test script for the uniform-thickness skeletonization feature.

This script demonstrates the difference between standard TPS-Inbetween processing
and uniform-thickness processing using actual test images.
"""

import os
import sys
import tempfile
import shutil

# Add the wrappers directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
wrappers_dir = os.path.join(current_dir, 'wrappers')
sys.path.append(wrappers_dir)

from tps_inbetween_wrapper import TPSInbetweenWrapper


def test_uniform_thickness():
    """Test the uniform thickness feature with sample images."""
    
    # Use two consecutive frames from the cleaned images
    img0_path = "../cleaned_images/ep01_sc324_Cyberslav_CLEAN_0007.jpg"
    img1_path = "../cleaned_images/ep01_sc324_Cyberslav_CLEAN_0008.jpg"
    
    # Check if test images exist
    if not os.path.exists(img0_path):
        print(f"❌ Test image not found: {img0_path}")
        return False
    
    if not os.path.exists(img1_path):
        print(f"❌ Test image not found: {img1_path}")
        return False
    
    print("🧪 Testing Uniform Thickness Feature")
    print("=" * 50)
    print(f"Input 0: {img0_path}")
    print(f"Input 1: {img1_path}")
    print()
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Test 1: Standard processing (no uniform thickness)
        print("\n🔍 Test 1: Standard Processing")
        print("-" * 30)
        
        wrapper_standard = TPSInbetweenWrapper(
            use_cpu=True,
            uniform_thin=False
        )
        
        try:
            output_standard = wrapper_standard.interpolate(
                img0_path=img0_path,
                img1_path=img1_path,
                output_path=os.path.join(temp_dir, "standard_output.png"),
                temp_dir=os.path.join(temp_dir, "standard_temp"),
                create_gif=True
            )
            print(f"✅ Standard output: {output_standard}")
        except Exception as e:
            print(f"❌ Standard processing failed: {e}")
            return False
        
        # Test 2: Uniform thickness processing
        print("\n🔍 Test 2: Uniform Thickness Processing")
        print("-" * 30)
        
        wrapper_uniform = TPSInbetweenWrapper(
            use_cpu=True,
            uniform_thin=True
        )
        
        try:
            output_uniform = wrapper_uniform.interpolate(
                img0_path=img0_path,
                img1_path=img1_path,
                output_path=os.path.join(temp_dir, "uniform_output.png"),
                temp_dir=os.path.join(temp_dir, "uniform_temp"),
                create_gif=True
            )
            print(f"✅ Uniform thickness output: {output_uniform}")
        except Exception as e:
            print(f"❌ Uniform thickness processing failed: {e}")
            return False
        
        # Copy results to current directory for inspection
        results_dir = "uniform_thickness_test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Copy standard results
        if os.path.exists(output_standard):
            shutil.copy2(output_standard, os.path.join(results_dir, "standard_output.png"))
            gif_standard = output_standard.replace('.png', '.gif')
            if os.path.exists(gif_standard):
                shutil.copy2(gif_standard, os.path.join(results_dir, "standard_output.gif"))
        
        # Copy uniform thickness results
        if os.path.exists(output_uniform):
            shutil.copy2(output_uniform, os.path.join(results_dir, "uniform_output.png"))
            gif_uniform = output_uniform.replace('.png', '.gif')
            if os.path.exists(gif_uniform):
                shutil.copy2(gif_uniform, os.path.join(results_dir, "uniform_output.gif"))
        
        print(f"\n📁 Results saved to: {results_dir}/")
        print("   • standard_output.png - Standard processing result")
        print("   • uniform_output.png - Uniform thickness result")
        print("   • standard_output.gif - Standard processing GIF")
        print("   • uniform_output.gif - Uniform thickness GIF")
        
        print("\n✅ Both tests completed successfully!")
        print("\n📝 Comparison:")
        print("   • Standard processing: May show thickness variations")
        print("   • Uniform thickness: Consistent stroke width throughout")
        
        return True


if __name__ == '__main__':
    success = test_uniform_thickness()
    if success:
        print("\n🎉 Uniform thickness feature test passed!")
    else:
        print("\n💥 Uniform thickness feature test failed!")
        sys.exit(1) 