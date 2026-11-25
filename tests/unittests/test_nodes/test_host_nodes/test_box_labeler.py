from datetime import timedelta

import depthai as dai
import numpy as np

from depthai_nodes.message import BoxDetection, BoxDetections
from depthai_nodes.node.box_associator.box_labeler import BoxLabeler
from tests.utils import OutputMock

DEPTH_SIZE = (8, 6)


def detection(track_id: int) -> BoxDetection:
    return BoxDetection(
        track_id=track_id,
        bounding_box=dai.RotatedRect(dai.Rect(dai.Point2f(0, 0), dai.Point2f(1, 1))),
        confidence=0.9,
        track_status=dai.Tracklet.TrackingStatus.TRACKED,
    )


def create_depth(size: tuple[int, int] = DEPTH_SIZE) -> dai.ImgFrame:
    width, height = size
    depth = dai.ImgFrame()
    depth.setCvFrame(np.full((height, width), 1000, np.uint16), dai.ImgFrame.Type.RAW16)
    return depth


def label(detections: BoxDetections, depth: dai.ImgFrame, view: int) -> dai.ImgFrame:
    labeler = BoxLabeler().build(OutputMock(), OutputMock(), view=view)
    queue = labeler.out.createOutputQueue()
    labeler.process(detections, depth)
    image = queue.get()
    assert isinstance(image, dai.ImgFrame)
    return image


def test_each_box_is_painted_with_its_track_bytes_and_view():
    mask = np.full((6, 8), 255, np.uint8)
    mask[1:3, 1:4] = 0
    mask[4:6, 5:8] = 1
    detections = BoxDetections([detection(0x0102), detection(70_000)], mask)

    image = label(detections, create_depth(), view=2)

    pixels = image.getFrame()
    expected = np.zeros((6, 8, 3), np.uint8)
    expected[1:3, 1:4] = (0x02, 0x01, 3)
    expected[4:6, 5:8] = (0x70, 0x11, 3)
    np.testing.assert_array_equal(pixels, expected)


def test_a_mask_at_another_resolution_is_mapped_onto_the_depth_by_nearest_neighbour():
    mask = np.array([[0, 255, 255, 1], [255, 0, 1, 255]], np.uint8)
    detections = BoxDetections([detection(5), detection(6)], mask)

    image = label(detections, create_depth(), view=0)

    assert image.getType() == dai.ImgFrame.Type.RGB888i
    assert (image.getWidth(), image.getHeight()) == DEPTH_SIZE
    tracks = image.getFrame()[:, :, 0]
    expected_mask = np.repeat(np.repeat(mask, 3, axis=0), 2, axis=1)
    expected = np.select([expected_mask == 0, expected_mask == 1], [5, 6], 0)
    np.testing.assert_array_equal(tracks, expected)


def stamp(message: dai.Buffer, seconds: int, sequence: int) -> None:
    message.setTimestamp(timedelta(seconds=seconds))
    message.setTimestampDevice(timedelta(seconds=seconds + 10))
    message.setSequenceNum(sequence)


def assert_stamped(message: dai.Buffer, seconds: int, sequence: int) -> None:
    assert message.getTimestamp() == timedelta(seconds=seconds)
    assert message.getTimestampDevice() == timedelta(seconds=seconds + 10)
    assert message.getSequenceNum() == sequence


def test_the_label_image_carries_the_depth_metadata():
    depth = create_depth()
    stamp(depth, seconds=3, sequence=42)
    transformation = dai.ImgTransformation(*DEPTH_SIZE)
    transformation.setIntrinsicMatrix([[7.0, 0.0, 4.0], [0.0, 7.0, 3.0], [0, 0, 1]])
    depth.setTransformation(transformation)
    detections = BoxDetections([], np.full((6, 8), 255, np.uint8))
    stamp(detections, seconds=1, sequence=7)

    image = label(detections, depth, view=0)

    assert_stamped(image, seconds=3, sequence=42)
    assert image.getTransformation().getIntrinsicMatrix() == (
        transformation.getIntrinsicMatrix()
    )


def test_passthrough_detections_carry_the_depth_stamps_and_leave_the_input_alone():
    mask = np.full((3, 4), 255, np.uint8)
    mask[1, 2] = 0
    items = [detection(9)]
    detections = BoxDetections(items, mask)
    stamp(detections, seconds=1, sequence=7)
    depth = create_depth()
    stamp(depth, seconds=3, sequence=42)
    detections_out, depth_out = OutputMock(), OutputMock()
    labeler = BoxLabeler().build(detections_out, depth_out, view=0)
    queue = labeler.passthroughDetections.createOutputQueue()

    labeler.process(detections, depth)

    passthrough = queue.get()
    assert labeler._linked_args == (detections_out, depth_out)
    assert isinstance(passthrough, BoxDetections)
    assert passthrough is not detections
    assert passthrough.items == items
    np.testing.assert_array_equal(passthrough.segmentation_mask, mask)
    assert_stamped(passthrough, seconds=3, sequence=42)
    assert_stamped(detections, seconds=1, sequence=7)
