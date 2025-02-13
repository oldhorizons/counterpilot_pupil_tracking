from picamzero import Camera
import numpy as np
import requests
import time
from base64 import b64encode
import json

url = "192.168.0.72:8080"

def send_request(image):
    xdim, ydim = image.shape
    encoded_image = b64encode(image).decode('utf-8')
    response = requests.post(url, json={"image": encoded_image, "xdim": xdim, "ydim": ydim})
    return response

cam = Camera()

while True:
    img = cam.capture_array()
    response = send_request(img)
    time.sleep(10)