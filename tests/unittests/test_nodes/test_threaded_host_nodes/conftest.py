from pytest import Config

from tests.utils import HostNodeMock, PipelineMock, ThreadedHostNodeMock


def pytest_configure(config: Config):
    import depthai as dai

    dai.Pipeline = PipelineMock
    dai.node.ThreadedHostNode = ThreadedHostNodeMock
    dai.node.HostNode = HostNodeMock
