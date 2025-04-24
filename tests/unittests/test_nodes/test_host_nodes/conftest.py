from tests.utils import HostNodeMock


def pytest_configure():
    import depthai as dai

    dai.node.HostNode = HostNodeMock
