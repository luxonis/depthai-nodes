# Message Types

Here are the custom message types that we introduce in this package. They are used as output types of the parsers.

**Table of Contents**

- [Message Types](#message-types)
  - [Classifications](#classifications)
    - [Attributes](#attributes)
  - [Cluster](#cluster)
    - [Attributes](#attributes-1)
  - [Clusters](#clusters)
    - [Attributes](#attributes-2)
  - [Collection](#collection)
    - [Attributes](#attributes-3)
  - [GatheredData](#gathereddata)
    - [Attributes](#attributes-4)
  - [Keypoints](#keypoints)
    - [Attributes](#attributes-5)
  - [Line](#line)
    - [Attributes](#attributes-6)
  - [Lines](#lines)
    - [Attributes](#attributes-7)
  - [Map2D](#map2d)
    - [Attributes](#attributes-8)
  - [Prediction](#prediction)
    - [Attributes](#attributes-9)
  - [Predictions](#predictions)
    - [Attributes](#attributes-10)
  - [SnapData](#snapdata)
    - [Attributes](#attributes-12)

## Classifications

Classification class for storing the classes and their respective scores.

### Attributes

- **classes** (list\[str\]): A list of classes.
- **scores** (NDArray\[np.float32\]): Corresponding probability scores.

## Cluster

Cluster class for storing a cluster.

### Attributes

- **label** (int): Label of the cluster.
- **points** (List\[dai.Point2f\]): List of points in the cluster.

## Clusters

Clusters class for storing clusters.

### Attributes

- **clusters** (List\[[Cluster](#cluster)\]): List of clusters.

## Collection

Collection class for storing a list of messages or other items of the same type.

### Attributes

- **items** (List\[T\]): List of collected items.
- **item_cls** (Optional\[Type\[T\]\]): Runtime item type inferred from the first item once the collection is non-empty.

Items can be added with `append(...)` or `extend(...)`. The collection enforces that all items have the same inferred type.

## GatheredData

GatheredData class for storing a reference message and the messages gathered for that reference.

### Attributes

- **reference_data** (TReference): Reference message used to determine how many items to gather.
- **items** (List\[TGathered\]): List of gathered messages.
- **item_cls** (Optional\[Type\[TGathered\]\]): Runtime gathered-item type inferred from the first item once the collection is non-empty.

GatheredData inherits [Collection](#collection) behavior, so all gathered items must have the same inferred type. Setting `reference_data` copies the reference message sequence number, timestamp, and device timestamp to the GatheredData message.

## Keypoints

Keypoints class for storing keypoints and optional skeleton edges.

### Attributes

- **keypoints_list** (dai.KeypointsList): Native DepthAI keypoints list containing keypoints and edges.
- **transformation** (Optional\[dai.ImgTransformation\]): Optional image transformation associated with the keypoints.

The keypoints can be accessed with `getKeypoints()` and set with `setKeypoints(...)`. Each keypoint is a `dai.Keypoint` with image coordinates, confidence, and optional label name. Edges can be accessed with `getEdges()` and set with `setEdges(...)`.

## Line

Line class for storing a line.

### Attributes

- **start_point** (dai.Point2f): Start point of the line with x and y coordinates.
- **end_point** (dai.Point2f): End point of the line with x and y coordinates.
- **confidence** (float): Confidence of the line.

## Lines

Lines class for storing lines.

### Attributes

- **lines** (List\[[Line](#line)\]): List of detected lines.

## Map2D

Map2D class for storing a 2D map of floats.

### Attributes

- **map** (NDArray\[np.float32\]): 2D map.
- **width** (int): 2D Map width.
- **height** (int): 2D Map height.

## Prediction

Prediction class for storing a prediction.

### Attributes

- **prediction** (float): The predicted value.

## Predictions

Predictions class for storing predictions.

### Attributes

- **predictions** (List\[[Prediction](#prediction)\]): List of predictions.

## SnapData

SnapData class for representing a single snap event to be uploaded to DepthAI Hub.

### Attributes

- **snap_name** (str): Logical name of the snap.
- **file_group** (dai.FileGroup): Object containing the snap image and associated data (e.g., images, detections).
- **tags** (List\[str\]): Optional list of tags for categorizing the snap.
- **extras** (Dict\[str, str\]): Additional metadata as key-value pairs.
