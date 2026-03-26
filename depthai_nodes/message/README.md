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
  - [Line](#line)
    - [Attributes](#attributes-7)
  - [Lines](#lines)
    - [Attributes](#attributes-8)
  - [Map2D](#map2d)
    - [Attributes](#attributes-9)
  - [Prediction](#prediction)
    - [Attributes](#attributes-10)
  - [Predictions](#predictions)
    - [Attributes](#attributes-11)
  - [SegmentationMask](#segmentationmask)
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

## SegmentationMask

SegmentationMask class for a single- or multi-object segmentation mask. Background is represented with "-1" and foreground classes with non-negative integers.

### Attributes

- **mask** (NDArray\[np.int16\]): Segmentation mask.
