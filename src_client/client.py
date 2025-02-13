from picamzero import Camera
import numpy as np
import requests
import time
from base64 import b64encode
import json
from pythonosc import udp_client, osc_message
import constants

ip = constants.ip
port = constants.http_port
addr = "http://" + ip + ":" + str(port)

def send_osc(image):
    print(image.shape)
    xdim, ydim, _ = image.shape
    encoded_image = b64encode(image).decode('utf-8')
    message = osc_message.OscMessage(encoded_image)
    client.sent_message("/image", message)
    time.sleep(1)
   
def send_request(image):
    xdim, ydim, _ = image.shape
    encoded_image = b64encode(image).decode('utf-8')
    response = requests.post(addr, headers={"Host": "www.google.com"}, json={"image": encoded_image, "xdim": xdim, "ydim": ydim})
    return response

cam = Camera()
client = udp_client.SimpleUDPClient(ip, port)

while True:
    img = cam.capture_array()
    #cam.take_photo("beep.jpg")
    send_request(img)
    time.sleep(1)
