import base64
import numpy as np
import json
import cv2
import pypupilext as pp
from matplotlib import pyplot as plt
from functools import cached_property
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qsl, urlparse

model = pp.Starburst()
x, y, w, h = 0, 0, 0, 0

def process_request(json):
    #extract data
    byteData = json['image']
    xDim = json['xdim']
    yDim = json['ydim']

    #convert json back to readable image encoding
    nums = base64.decodebytes(byteData)
    npArray = np.frombuffer(nums, dtype=np.float64)
    np.reshape(npArray, (xDim, yDim))
    img = npArray.astype('uint8')
    plt.imshow(img)
    plt.show()

    # if ROI not initialised, initialise
    if w == 0:
        x, y, w, h = cv2.selectROI(img)
    cv2.destroyAllWindows()
    img_crop = img[y:y+h, x:x+w]
    cv2.imshow("cropped", img_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    grayscale = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    pupil = model.run(grayscale)
    print(pupil)


class TrackerServer(BaseHTTPRequestHandler):
    def post_data(self):
        content_length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(content_length)

    def do_GET(self):
        print("MESSAGE RECEIVED")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = process_request(self.post_data())
        self.wfile.write(response.encode("utf-8"))
        print("GOTTEM")

    def do_POST(self):
        self.do_GET()

if __name__ == "__main__":
    print("starting")
    server = HTTPServer(("192.168.0.72", 5005), TrackerServer)
    print("server initialised")
    server.serve_forever()
    print("serving on 192.68.0.72:5005")
    