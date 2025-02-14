import platform
ip = "192.168.0.72"
if platform.system() == "Linux": #VM, only has access to local network
    host_internal_ip = "192.168.0.92"
else:
    host_internal_ip = ip
http_port = 5005
osc_server_port = 5006
osc_client_port = 5007
