import inspect

import pytest

from depthai_nodes.node.box_associator.box_cloud_assembler import BoxCloudAssembler
from depthai_nodes.node.box_associator.box_labeler import BoxLabeler


@pytest.mark.parametrize("node", [BoxLabeler, BoxCloudAssembler])
def test_process_annotations_are_types_for_link_args(node):
    # depthai's HostNode.link_args reads each process annotation's __name__, so a
    # string annotation (e.g. from `from __future__ import annotations`) crashes build.
    parameters = inspect.signature(node.process).parameters.values()
    assert not [p.name for p in parameters if isinstance(p.annotation, str)]
