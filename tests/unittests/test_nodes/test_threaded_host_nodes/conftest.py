from tests.utils import HostNodeMock, PipelineMock, SyncMock, ThreadedHostNodeMock


def pytest_configure():
    import depthai as dai

    dai.Pipeline = PipelineMock
    dai.node.ThreadedHostNode = ThreadedHostNodeMock
    dai.node.HostNode = HostNodeMock
    dai.node.Sync = SyncMock
