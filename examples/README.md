# DepthAI Nodes examples

The `main.py` script lets you run fully-automated pipeline with the model of your choice. To run the script:

Make sure you have the `depthai-nodes` package installed:

```
cd depthai-nodes
pip install -e .
```

Prepare the model slug and run:

```
cd examples
python main.py -s <slug_of_your_model>
```

For example:

```
python main.py -s yolov6-nano:coco-416x416
```

Note that for running the examples you need RVC2 device connected.
For now, only RVC2 models can be used.
If using OAK-D Lite, make sure to also set the FPS limit under 28.5.

For example:

```
python main.py -s yolov6-nano:coco-416x416 -fps 28
```

Some models have small input sizes and requesting small image size from `Camera` is problematic so we request 4x bigger frame and resize it back down. During visualization image frame is resized back so some image quality is lost - only for visualization.

The parser is obtained from NN archive along with other important parameters for the parser. So, make sure your NN archive is well-defined.

### XFeat

If you want to run xfeat demo you have two options available - to run it in `Stereo` mode or `Mono` mode depending on the nn archive you provided. If the NN archive requires `XFeatMonoParser` then the mono mode will be used, otherwise the stereo mode will be used (`XFeatStereoParser`). For the stereo mode you need OAK camera which has left and right cameras, if not the error will be raised. If you use mono mode you can set the reference frame to which all the other frames will be compared to. The reference frame is set by triggering - pressing `S` key.
