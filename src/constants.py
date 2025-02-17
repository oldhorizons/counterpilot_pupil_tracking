import platform
ip = "192.168.0.72"
if platform.system() == "Linux":
    host_internal_ip = "192.168.0.92"
else:
    host_internal_ip = ip
osc_ip = "10.0.0.123"
http_port = 5005
osc_port = 5006
