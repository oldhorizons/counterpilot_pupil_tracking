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
from test_server_osc import OSCCommunicator
from constants import ip, http_port, osc_server_port, osc_client_port
from queue import Queue
import io

class TrackerRequestHandler(BaseHTTPRequestHandler):
    oscCommunicator = OSCCommunicator(ip, osc_server_port, osc_client_port)
    model = pp.Starburst()
    dims = (0, 0, 0, 0) #x, y, w, h

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def draw_pupil_and_show(self, cv2Image, pupil):
        x, y = pupil.center
        d = pupil.physicalDiameter
        if d < 0:
            print("no pupil found")
            return
        colourImg = cv2.cvtColor(cv2Image, cv2.COLOR_GRAY2RGB)
        #draw circumference
        cv2.circle(colourImg, (x, y), d//2, (255, 0, 0), 1)
        #draw center
        cv2.circle(colourImg, (x, y), 1, (0, 255, 0), 1) 
        # resize = cv2.resize(cv2Image, None, fx=scale, fy=scale)
        cv2.imshow("label", cv2Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_request(self, json):
        #extract data
        json = json.decode("utf-8")
        if not (json[0] == "{" and json[-1] == "}" and json.count('}') == 1):
            print("Invalid request response. Request bodies should be a single json dict with no subdicts.")
            return
        json = eval(json)
        byteData = json['image']
        xDim = json['xdim']
        yDim = json['ydim']

        #convert json back to readable image encoding
        decoded_image = base64.b64decode(byteData)
        image_bytes_io = io.BytesIO(decoded_image)
        npImg = np.frombuffer(image_bytes_io.getvalue(), np.uint8)
        img = npImg.reshape(xDim, yDim, 3)

        # if ROI not initialised, initialise
        if self.dims == (0,0,0,0):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Select Area of Interest", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Select Area of Interest", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.dims = cv2.selectROI("Select Area of Interest", img)
            x, y, w, h = self.dims
            cv2.destroyAllWindows()
            img_crop = img[y:y+h, x:x+w]
            cv2.imshow("cropped", img_crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        grayscale = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        pupil = self.model.run(grayscale)
        self.draw_pupil_and_show(grayscale, pupil)
        x, y, w, h  = self.dims
        # self.oscCommunicator.actionableQueue.put(pupil)
        return f"[{x}, {y}, {w}, {h}]"
    
    def post_data(self):
        content_length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(content_length)

    def do_GET(self):
        print("MESSAGE RECEIVED")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = self.process_request(self.post_data())
        self.wfile.write(response.encode("utf-8"))
        print("sent response")

    def do_POST(self):
        self.do_GET()

if __name__ == "__main__":
    print("starting")
    server = HTTPServer((ip, http_port), TrackerRequestHandler)
    print(f"serving on {ip}:{http_port}")
    server.serve_forever()
    