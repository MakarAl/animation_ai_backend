import requests
import time
import os
import sys

# Add the parent directory to get sample images
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_full_api_flow():
    """Test the complete API flow with real images."""
    base_url = "http://127.0.0.1:8000"
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            print("❌ Server is not running or not healthy")
            return False
        print("✅ Server is healthy")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please start the server with: python -m uvicorn api.main:app --host 127.0.0.1 --port 8000")
        return False
    
    # Find sample images
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input_images')
    if not os.path.exists(sample_dir):
        print(f"❌ Sample images directory not found: {sample_dir}")
        return False
    
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(sample_files) < 2:
        print(f"❌ Need at least 2 sample images, found {len(sample_files)}")
        return False
    
    img0_path = os.path.join(sample_dir, sample_files[0])
    img1_path = os.path.join(sample_dir, sample_files[1])
    
    print(f"📸 Using sample images: {sample_files[0]} and {sample_files[1]}")
    
    # Step 1: Create interpolation job
    print("\n🔄 Step 1: Creating interpolation job...")
    try:
        with open(img0_path, 'rb') as f0, open(img1_path, 'rb') as f1:
            files = {
                'keyframe_0': (sample_files[0], f0, 'image/jpeg'),
                'keyframe_1': (sample_files[1], f1, 'image/jpeg')
            }
            data = {
                'size': 512,  # Smaller size for faster testing
                'vector_cleanup': False,
                'no_edge_sharpen': True,
                'uniform_thin': False
            }
            
            response = requests.post(f"{base_url}/api/v1/interpolate", files=files, data=data)
            
        if response.status_code != 200:
            print(f"❌ Failed to create job: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        job_data = response.json()
        job_id = job_data['job_id']
        print(f"✅ Job created successfully: {job_id}")
        
    except Exception as e:
        print(f"❌ Error creating job: {e}")
        return False
    
    # Step 2: Monitor job status
    print("\n🔄 Step 2: Monitoring job status...")
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"{base_url}/api/v1/jobs/{job_id}")
            if response.status_code != 200:
                print(f"❌ Failed to get job status: {response.status_code}")
                return False
            
            job_status = response.json()
            status = job_status['status']
            print(f"📊 Job status: {status}")
            
            if status == "COMPLETED":
                print("✅ Job completed successfully!")
                result_url = job_status.get('result_url')
                if result_url:
                    print(f"📁 Result file: {result_url}")
                break
            elif status == "FAILED":
                error_msg = job_status.get('error_message', 'Unknown error')
                print(f"❌ Job failed: {error_msg}")
                return False
            elif status in ["PENDING", "PROCESSING"]:
                print("⏳ Job still processing...")
                time.sleep(5)  # Wait 5 seconds before checking again
            else:
                print(f"❌ Unknown job status: {status}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking job status: {e}")
            return False
    
    if time.time() - start_time >= max_wait_time:
        print("❌ Job timed out")
        return False
    
    # Step 3: Download result
    print("\n🔄 Step 3: Downloading result...")
    try:
        response = requests.get(f"{base_url}/api/v1/jobs/{job_id}/result")
        if response.status_code != 200:
            print(f"❌ Failed to download result: {response.status_code}")
            return False
        
        # Save result
        result_path = f"temp/api_test_result_{job_id}.png"
        os.makedirs("temp", exist_ok=True)
        with open(result_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Result downloaded successfully: {result_path}")
        
        # Check file size
        file_size = os.path.getsize(result_path)
        print(f"📏 Result file size: {file_size} bytes")
        
        if file_size > 0:
            print("✅ Result file is valid")
            return True
        else:
            print("❌ Result file is empty")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading result: {e}")
        return False

if __name__ == "__main__":
    print("Testing Full API Flow with TPS Integration")
    print("=" * 60)
    
    success = test_full_api_flow()
    
    print("=" * 60)
    if success:
        print("🎉 Full API flow test completed successfully!")
        print("The backend is ready for production use.")
    else:
        print("❌ Full API flow test failed.")
        print("Please check the error messages above.")
