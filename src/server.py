from http.server import BaseHTTPRequestHandler
import imageProcessing
import base64
import numpy as np
import json

# webapp.py

from functools import cached_property
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qsl, urlparse


def process_request(json):
    #extract data
    byteData = json['image']
    xDim = json['xdim']
    yDim = json['ydim']

    #convert json back to readable image encoding
    nums = base64.decodebytes(byteData)
    npArray = np.frombuffer(nums, dtype=np.float64)
    np.reshape(npArray, (xDim, yDim))
    image = npArray.astype('uint8')

class TrackerServer(BaseHTTPRequestHandler):
    @cached_property
    def url(self):
        return urlparse(self.path)

    @cached_property
    def query_data(self):
        return dict(parse_qsl(self.url.query))

    @cached_property
    def post_data(self):
        content_length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(content_length)

    @cached_property
    def form_data(self):
        return dict(parse_qsl(self.post_data.decode("utf-8")))

    @cached_property
    def cookies(self):
        return SimpleCookie(self.headers.get("Cookie"))

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.process_request().encode("utf-8"))

    def do_POST(self):
        self.do_GET()


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8000), TrackerServer)
    server.serve_forever()
    