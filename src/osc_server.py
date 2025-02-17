import argparse
import random
import time
import threading
from queue import Queue

from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
import constants
# https://github.com/attwad/python-osc/blob/main/examples/simple_2way.py
ip = constants.ip
server_port = constants.osc_server_port
client_port = constants.osc_client_port

class OSCCommunicator:
    def __init__(self, ip, server_port, client_port):
        self.ip = ip
        self.server_port = server_port
        self.client_port = client_port
        # listen to addresses and print changes in values
        self.dispatcher = Dispatcher()
        self.dispatcher.map("/1/echo", self.server_echo_handler, "Echo")
        self.dispatcher.map("/ping", print)
        self.dispatcher.map("/message", self.server_message_handler, "Messager")
        self.actionableQueue = Queue()
        self.messageQueue = Queue()
      
    def server_echo_handler(self, unused_addr, args, value):
        print("received: " + args)
        print(value)
    
    def server_message_handler(self, unused_addr, args, value):
        self.actionableQueue.put(args)
        
    def start_server(self):
        print("Starting Server")
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.server_port), self.dispatcher)
        print(f"Serving on {self.server.server_address}")
        thread = threading.Thread(target=self.server.serve_forever)
        thread.start()
    
    def start_client(self):
        print("Starting Client")
        self.client = udp_client.SimpleUDPClient(self.ip, self.client_port)
        # print("Sending on {}".format(client.))
        thread = threading.Thread(target=self.client_message_handler("hello world!", "/1/echo"))
        thread.start()

    def client_message_handler(self, message, endpoint):
        while True:
            endpoint, message = self.messageQueue.get()
            self.client.send_message(endpoint, message)
            self.messageQueue.task_done()

    def start(self):
        self.start_server()
        self.start_client()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serverip", default=ip, help="The ip to listen on")
    parser.add_argument(
        "--serverport",
        type=int,
        default=server_port,
        help="The port the OSC Server is listening on",
    )
    parser.add_argument(
        "--clientip", default=ip, help="The ip of the OSC server"
    )
    parser.add_argument(
        "--clientport",
        type=int,
        default=client_port,
        help="The port the OSC Client is listening on",
    )
    args = parser.parse_args()
