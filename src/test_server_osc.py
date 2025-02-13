import argparse
import random
import time
import threading

from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
import constants
ip = constants.ip
server_port = constants.osc_server_port
client_port = constants.osc_client_port


def print_fader_handler(unused_addr, args, value):
    print(f"[{args[0]}] ~ {value:0.2f}")


def print_xy_fader_handler(unused_addr, args, value1, value2):
    print(f"[{args[0]}] ~ {value2:0.2f} ~ {value1:0.2f}")


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

    # listen to addresses and print changes in values
    dispatcher = Dispatcher()
    dispatcher.map("/1/push2", print)
    dispatcher.map("/1/fader1", print_fader_handler, "Focus")
    dispatcher.map("/1/fader2", print_fader_handler, "Zoom")
    dispatcher.map("/1/xy1", print_xy_fader_handler, "Pan-Tilt")
    dispatcher.map("/ping", print)


def start_server(ip, port):
    print("Starting Server")
    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print(f"Serving on {server.server_address}")
    thread = threading.Thread(target=server.serve_forever)
    thread.start()


def start_client(ip, port):
    print("Starting Client")
    client = udp_client.SimpleUDPClient(ip, port)
    # print("Sending on {}".format(client.))
    thread = threading.Thread(target=random_values(client))
    thread.start()


# send random values between 0-1 to the three addresses
def random_values(client):
    while True:
        for x in range(10):
            client.send_message("/1/fader2", random.random())
            client.send_message("/1/fader1", random.random())
            client.send_message("/1/xy1", [random.random(), random.random()])
            time.sleep(0.5)


start_server(args.serverip, args.serverport)
start_client(args.clientip, args.clientport)