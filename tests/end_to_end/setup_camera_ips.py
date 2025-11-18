import platform

import depthai as dai

devices = dai.Device.getAllAvailableDevices()
RVC2_IP = ""
RVC4_IP = ""
for device in devices:
    mxid = device.getDeviceId()
    dev_platform = device.platform.name
    if "RVC4" in dev_platform:
        RVC4_IP = str(mxid)
    if "MYRIAD" in dev_platform:
        RVC2_IP = str(mxid)

system = platform.system().lower()

if system == "windows":
    # For Windows CMD
    print(f"set RVC2_IP={RVC2_IP}")
    print(f"set RVC4_IP={RVC4_IP}")
else:
    # For Linux / macOS / others
    print(f"export RVC2_IP={RVC2_IP}")
    print(f"export RVC4_IP={RVC4_IP}")
