from picamzero import Camera
import numpy as np
import requests
import time
from base64 import b64encode
import json
from pythonosc import udp_client, osc_message

ip = "192.168.0.92"
port = 5005
addr = "http://" + ip + ":" + str(port)
x, y, w, h = 0, 0, 0, 0

def send_osc(image):
    print(image.shape)
    xdim, ydim, _ = image.shape
    encoded_image = b64encode(image).decode('utf-8')
    message = osc_message.OscMessage(encoded_image)
    client.send_message("/image", message)
    time.sleep(1)
    
def send_request(image):
    print(image.shape)
    xdim, ydim, _ = image.shape
    encoded_image = b64encode(image).decode('utf-8')
    print("sending request")
    response = requests.post(addr, headers={"Host": "www.google.com"}, json={"image": encoded_image, "xdim": xdim, "ydim": ydim})
    print(response)
    return response

cam = Camera()
client = udp_client.SimpleUDPClient(ip, port)
print("setup complete")

while True:
    img = cam.capture_array()
    print("image captured")
    send_request(img)
    time.sleep(1)
