from typing import List

import cv2
import depthai as dai
from visualization.visualizers import xfeat_visualizer

from depthai_nodes.ml.parsers import XFeatMonoParser, XFeatStereoParser


def xfeat_mono(nn_archive: dai.NNArchive, input_shape: List[int], fps_limit: int):
    """Run the XFeatMonoParser on a single camera.

    It lets you set the reference frame by pressing S-key.
    """
    previous_frame = None
    with dai.Pipeline() as pipeline:
        # Set up camera
        cam = pipeline.create(dai.node.Camera).build()

        # Set up the neural network
        network = pipeline.create(dai.node.NeuralNetwork).build(
            cam.requestOutput(
                input_shape, type=dai.ImgFrame.Type.BGR888p, fps=fps_limit
            ),
            nn_archive,
        )

        # Set up parser
        parser = XFeatMonoParser()
        parser.setOriginalSize(input_shape)
        parser.setInputSize(input_shape)
        parser.setMaxKeypoints(2048)

        # Linking
        network.out.link(parser.input)

        # Set up queue
        camera_queue = network.passthrough.createOutputQueue()
        parser_queue = parser.out.createOutputQueue()

        pipeline.start()

        while pipeline.isRunning():
            frame: dai.ImgFrame = camera_queue.get().getCvFrame()
            message: dai.TrackedFeatures = (
                parser_queue.get()
            )  # get message from the queue
            features = message.trackedFeatures
            if previous_frame is not None:
                resulting_frame = xfeat_visualizer(previous_frame, frame, features)
            else:
                resulting_frame = frame
            number_of_matches = len(features) // 2
            cv2.putText(
                resulting_frame,
                f"Number of matches: {number_of_matches}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("XFeat", resulting_frame)

            key_pressed = cv2.waitKey(1)
            if key_pressed == ord("s"):
                parser.setTrigger()  # trigger to set the reference frame
                previous_frame = frame
            if key_pressed == ord("q"):
                cv2.destroyAllWindows()
                pipeline.stop()
                break


def xfeat_stereo(nn_archive: dai.NNArchive, input_shape: List[int], fps_limit: int):
    """Run the XFeatStereoParser on stereo cameras - left and right - and match the features."""
    with dai.Pipeline() as pipeline:
        device: dai.Device = pipeline.getDefaultDevice()
        available_cameras = [
            camera.name for camera in device.getConnectedCameraFeatures()
        ]

        if "left" not in available_cameras or "right" not in available_cameras:
            raise RuntimeError(
                f"Stereo cameras are not available! Available cameras: {available_cameras}"
            )

        left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        left_network = pipeline.create(dai.node.NeuralNetwork).build(
            left_cam.requestOutput(
                input_shape, type=dai.ImgFrame.Type.RGB888p, fps=fps_limit
            ),
            nn_archive,
        )
        left_network.setNumInferenceThreads(2)

        right_network = pipeline.create(dai.node.NeuralNetwork).build(
            right_cam.requestOutput(
                input_shape, type=dai.ImgFrame.Type.RGB888p, fps=fps_limit
            ),
            nn_archive,
        )
        right_network.setNumInferenceThreads(2)

        parser = pipeline.create(XFeatStereoParser)
        parser.setOriginalSize(input_shape)
        parser.setInputSize(input_shape)
        parser.setMaxKeypoints(4096)

        left_network.out.link(parser.reference_input)
        right_network.out.link(parser.target_input)

        left_cam_queue = left_network.passthrough.createOutputQueue()
        right_cam_queue = right_network.passthrough.createOutputQueue()
        parser_queue = parser.out.createOutputQueue()

        pipeline.start()

        while pipeline.isRunning():
            left_frame: dai.ImgFrame = left_cam_queue.get().getCvFrame()
            right_frame: dai.ImgFrame = right_cam_queue.get().getCvFrame()
            features: dai.TrackedFeatures = parser_queue.get()
            features = features.trackedFeatures

            resulting_frame = xfeat_visualizer(left_frame, right_frame, features)
            number_of_matches = len(features) // 2
            cv2.putText(
                resulting_frame,
                f"Number of matches: {number_of_matches}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("XFeat Stereo", resulting_frame)

            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                pipeline.stop()
                break
