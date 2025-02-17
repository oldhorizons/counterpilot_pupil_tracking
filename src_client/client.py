from picamzero import Camera
import numpy as np
import requests
import time
from base64 import b64encode
from copy import deepcopy
#from matplotlib import pyplot as plt
import json
from pythonosc import udp_client, osc_message

class CaptureCommunicator:
	def __init__(self):
	#self.ip = "10.0.0.234"
		self.ip = "192.168.0.92"
		self.port = 5005
		self.addr = "http://" + self.ip + ":" + str(self.port)
		self.x, self.y, self.w, self.h = 0,0,0,0
		self.cam = Camera()
		self.cam.white_balance = 'cloudy'
		self.cam.flip_camera(vflip=True)
		self.client = udp_client.SimpleUDPClient(self.ip, self.port)
		print("setup complete")
		time.sleep(5)

	def send_osc(self, image):
		xdim, ydim, _ = image.shape
		encoded_image = b64encode(image).decode('utf-8')
		message = osc_message.OscMessage(encoded_image)
		self.client.send_message("/image", message)
		time.sleep(1)
		
	def send_request(self, image):
		xdim, ydim, _ = image.shape
		encoded_image = b64encode(image)
		utf8rep = encoded_image.decode('utf-8')
		print("sending request")
		response = requests.post(self.addr, headers={"Host": "www.google.com"}, json={"image": utf8rep, "xdim": xdim, "ydim": ydim})
		c = response.content.decode()
		if self.w == 0:
			self.x, self.y, self.w, self.h = [int(i) for i in response.content.decode()[1:-1].split(',')]
			print(f"ROI x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}")
		return response

	def run(self):
		while True:
			try:
				img = self.cam.capture_array()
				print("image captured")
				if self.w != 0:
					#todo persist if connection goes away
					img = deepcopy(img[self.y:self.y+self.h, self.x:self.x+self.w])
				#cam.take_photo("beep.jpg")
				self.send_request(img)
				time.sleep(5)
			except:
				print("dropped connection. Reestablishing.....")
				self.w = 0
				time.sleep(10)
				self.run()

if __name__ == "__main__":
	cc = CaptureCommunicator()
	cc.run()
