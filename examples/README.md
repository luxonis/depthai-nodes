# DepthAI Nodes examples

The `main.py` script lets you run fully-automated pipeline with the model of your choice. To run the script you need the model slug and then move to `examples` folder and run:

```
python main.py -s <slug_of_your_model>
```

For example:

```
python main.py -s yolov6-nano:coco-416x416
```

Note that for running the examples you need RVC2 device connected. For now, only RVC2 models can be used.

Some models have small input sizes and requesting small image size from `Camera` is problematic so we request 4x bigger frame and resize it back down. During visualization image frame is resized back so some image quality is lost - only for visualization.

The parser is obtained from NN archive along with other important parameters for the parser. So, make sure your NN archive is well-defined.