import depthai as dai
import pytest

from depthai_nodes.node.coordinates_mapper import CoordinatesMapper
from tests.utils import OutputMock, PipelineMock


class TransformationMessage:
    def __init__(self, transformation):
        self._transformation = transformation

    def getTransformation(self):
        return self._transformation


@pytest.fixture
def pipeline():
    pipeline = PipelineMock()
    pipeline._defaultDevice._platform = dai.Platform.RVC4
    return pipeline


@pytest.fixture
def mapper(pipeline: PipelineMock):
    return pipeline.create(CoordinatesMapper)


def test_extract_transformation_requires_valid_transformation(mapper: CoordinatesMapper):
    with pytest.raises(RuntimeError):
        mapper._extract_transformation(TransformationMessage(None))


def test_run_applies_new_transformation_before_next_source_message(
    mapper: CoordinatesMapper,
):
    first_transformation = object()
    updated_transformation = object()
    source_message_1 = dai.Buffer()
    source_message_2 = dai.Buffer()
    output = OutputMock()
    remap_calls = []
    running_states = iter([True, True, False])

    mapper._to_transformation_input.send(TransformationMessage(first_transformation))
    mapper._from_transformation_input.send(source_message_1)
    mapper._from_transformation_input.send(source_message_2)
    mapper._to_transformation_input.send(TransformationMessage(updated_transformation))
    mapper._out = output
    mapper.isRunning = lambda: next(running_states)

    def fake_remap_message(msg, to_transformation):
        remap_calls.append((msg, to_transformation))
        return msg

    mapper._remap_message = fake_remap_message

    mapper.run()

    assert remap_calls == [
        (source_message_1, updated_transformation),
        (source_message_2, updated_transformation),
    ]


def test_run_waits_for_first_transformation_and_sends_output(mapper: CoordinatesMapper):
    transformation = object()
    source_message = dai.Buffer()
    output = OutputMock()
    output_queue = output.createOutputQueue()
    running_states = iter([True, False])

    mapper._to_transformation_input.send(TransformationMessage(transformation))
    mapper._from_transformation_input.send(source_message)
    mapper._out = output
    mapper.isRunning = lambda: next(running_states)
    mapper._remap_message = lambda msg, to_transformation: (
        source_message if to_transformation is transformation else None
    )

    mapper.run()

    assert output_queue.get() is source_message
