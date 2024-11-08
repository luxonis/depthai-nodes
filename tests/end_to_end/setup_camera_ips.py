import os

import depthai as dai

devices = dai.Device.getAllAvailableDevices()
RVC2_IP = ""
RVC4_IP = ""
for device in devices:
    mxid = device.getMxId()
    platform = device.platform.name
    if "RVC4" in platform:
        RVC4_IP = str(mxid)
    if "MYRIAD" in platform:
        RVC2_IP = str(mxid)

# set env variables
os.environ["RVC2_IP"] = RVC2_IP
print(f"RVC2 IP/mxid: {os.getenv('RVC2_IP')}")
os.environ["RVC4_IP"] = RVC4_IP
print(f"RVC4 IP/mxid: {os.getenv('RVC4_IP')}")
