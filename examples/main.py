import depthai as dai
from utils.arguments import initialize_argparser, parse_fps_limit, parse_model_slug
from utils.model import get_input_shape, get_model_from_hub, get_parser
from utils.xfeat import xfeat_mono, xfeat_stereo
from visualization.visualize import visualize

from depthai_nodes.parser_generator import ParserGenerator

# Initialize the argument parser
arg_parser, args = initialize_argparser()

# Parse the model slug
model_slug, model_version_slug = parse_model_slug(args)
fps_limit = parse_fps_limit(args)

# Get the model from the HubAI
nn_archive = get_model_from_hub(model_slug, model_version_slug)

# Get the parser
parser_class, parser_name = get_parser(nn_archive)
input_shape = get_input_shape(nn_archive)

if parser_name == "XFeatMonoParser":
    xfeat_mono(nn_archive, input_shape, fps_limit)
    exit(0)
elif parser_name == "XFeatStereoParser":
    xfeat_stereo(nn_archive, input_shape, fps_limit)
    exit(0)

# Create the pipeline
with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()

    # YOLO and MobileNet-SSD have native parsers in DAI - no need to create a separate parser
    if parser_name == "YOLO" or parser_name == "SSD":
        network = pipeline.create(dai.node.DetectionNetwork).build(
            cam.requestOutput(
                input_shape, type=dai.ImgFrame.Type.BGR888p, fps=fps_limit
            ),
            nn_archive,
        )
        parser_queue = network.out.createOutputQueue()
    else:
        image_type = dai.ImgFrame.Type.BGR888p
        if "gray" in model_version_slug:
            image_type = dai.ImgFrame.Type.GRAY8

        if input_shape[0] < 128 or input_shape[1] < 128:
            print(
                "Input shape is too small so we are requesting a larger image and resizing it."
            )
            print(
                "During visualization we resize small image back to large, so image quality is lower."
            )
            manip = pipeline.create(dai.node.ImageManip)
            manip.initialConfig.setResize(input_shape)
            large_input_shape = (input_shape[0] * 4, input_shape[1] * 4)
            cam.requestOutput(large_input_shape, type=image_type, fps=fps_limit).link(
                manip.inputImage
            )
            network = pipeline.create(dai.node.NeuralNetwork).build(
                manip.out, nn_archive
            )
        else:
            network = pipeline.create(dai.node.NeuralNetwork).build(
                cam.requestOutput(input_shape, type=image_type, fps=fps_limit),
                nn_archive,
            )

        parsers = pipeline.create(ParserGenerator).build(nn_archive)

        if len(parsers) == 0:
            raise ValueError("No parsers were generated.")

        if len(parsers) > 1:
            raise ValueError("Only models with one parser are supported.")

        parser = parsers[0]

        # Linking
        network.out.link(parser.input)

        parser_queue = parser.out.createOutputQueue()

    camera_queue = network.passthrough.createOutputQueue()

    pipeline.start()

    while pipeline.isRunning():
        frame: dai.ImgFrame = camera_queue.get().getCvFrame()
        message = parser_queue.get()

        extraParams = nn_archive.getConfig().model.heads[0].metadata.extraParams
        if visualize(frame, message, parser_name, extraParams):
            pipeline.stop()
            break
