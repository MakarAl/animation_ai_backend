import requests
import os

# Test the API endpoints
base_url = "http://127.0.0.1:8000"

# Test root endpoint
print("Testing root endpoint...")
response = requests.get(f"{base_url}/")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test health endpoint
print("\nTesting health endpoint...")
response = requests.get(f"{base_url}/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test interpolation endpoint (without files for now)
print("\nTesting interpolation endpoint...")
try:
    response = requests.post(f"{base_url}/api/v1/interpolate")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

print("\nAPI documentation available at:")
print(f"{base_url}/docs")
