import requests

url = "http://127.0.0.1:5000/detect"
image_path = "demo2.jpeg"

files = {"image": open(image_path, "rb")}  # Open the image in binary mode
response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:", response.json())
