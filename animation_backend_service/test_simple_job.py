import requests
import json

def test_simple_job():
    base_url = "http://127.0.0.1:8000"
    
    # Test with minimal data
    try:
        files = {
            'keyframe_0': ('test1.jpg', b'fake_image_data', 'image/jpeg'),
            'keyframe_1': ('test2.jpg', b'fake_image_data', 'image/jpeg')
        }
        data = {
            'size': 512
        }
        
        print("Sending request...")
        response = requests.post(f"{base_url}/api/v1/interpolate", files=files, data=data)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_simple_job()
