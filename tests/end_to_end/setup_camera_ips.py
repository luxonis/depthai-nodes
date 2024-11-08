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
print(f"export RVC2_IP={RVC2_IP}")
print(f"export RVC4_IP={RVC4_IP}")
