import depthai as dai
from utils.arguments import initialize_argparser
from utils.model import get_input_shape, get_nn_archive_from_hub, get_parser
from utils.xfeat import xfeat_mono, xfeat_stereo
from visualization.visualize import visualize

from depthai_nodes import ParsingNeuralNetwork

# Initialize the argument parser
arg_parser, args = initialize_argparser()

# Parse the arguments
model = args.model
fps_limit = args.fps_limit

# Get the parser
nn_archive = get_nn_archive_from_hub(model)
_, parser_name = get_parser(nn_archive)
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

    if "gray" in model:
        image_type = dai.ImgFrame.Type.GRAY8
    else:
        image_type = dai.ImgFrame.Type.BGR888p

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
        nn = pipeline.create(ParsingNeuralNetwork).build(manip.out, model)
    else:
        nn = pipeline.create(ParsingNeuralNetwork).build(
            cam.requestOutput(input_shape, type=image_type, fps=fps_limit),
            model,
        )
    
    parser_queue = nn.out.createOutputQueue()

    camera_queue = nn.passthrough.createOutputQueue()

    pipeline.start()

    while pipeline.isRunning():
        frame: dai.ImgFrame = camera_queue.get().getCvFrame()
        message = parser_queue.get()

        extraParams = nn_archive.getConfig().model.heads[0].metadata.extraParams
        if visualize(frame, message, parser_name, extraParams):
            pipeline.stop()
            break
