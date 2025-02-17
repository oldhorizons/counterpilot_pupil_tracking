import base64
import numpy as np
import json
import pypupilext as pp
from matplotlib import pyplot as plt
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qsl, urlparse
from constants import ip, http_port, osc_ip, osc_port
import io
from pupil_tracker import PupilTracker

pupilTracker = PupilTracker(osc_ip, osc_port)

class TrackerRequestHandler(BaseHTTPRequestHandler):
    def process_request(self):
        global pupilTracker
        #extract data
        json = self.post_data()
        clientID = int(self.client_address[0].split('.')[-1])
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
        print("data extracted")
        x, y, w, h = pupilTracker.process_pupil(img, clientID)
        return f"[{x}, {y}, {w}, {h}]"

    def post_data(self):
        content_length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(content_length)

    def do_GET(self):
        print("MESSAGE RECEIVED")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = self.process_request()
        self.wfile.write(response.encode("utf-8"))
        print("sent response")

    def do_POST(self):
        self.do_GET()

if __name__ == "__main__":
    print("starting")
    server = HTTPServer((ip, http_port), TrackerRequestHandler)
    print(f"serving on {ip}:{http_port}")
    server.serve_forever()
    