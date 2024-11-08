import os

import depthai as dai

devices = dai.Device.getAllAvailableDevices()
for device in devices:
    mxid = device.getMxId()
    platform = device.platform.name
    RVC2_IP = ""
    RVC4_IP = ""
    if "RVC4" in platform:
        RVC4_IP = mxid
    if "MYRIAD" in platform:
        RVC2_IP = mxid

# set env variables
os.environ["RVC2_IP"] = RVC2_IP
os.environ["RVC4_IP"] = RVC4_IP
