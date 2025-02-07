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
  - [ImgDetectionExtended](#imgdetectionextended)
    - [Attributes](#attributes-3)
  - [ImgDetectionsExtended](#imgdetectionsextended)
    - [Attributes](#attributes-4)
  - [Keypoint](#keypoint)
    - [Attributes](#attributes-5)
  - [Keypoints](#keypoints)
    - [Attributes](#attributes-6)
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

## ImgDetectionExtended

A class for storing image detections in (x_center, y_center, width, height) format with additional angle and keypoints.

### Attributes

- **rotated_rect** (dai.RotatedRect): A depthai object for storing the roated bounding box information. The bounding box is stored as x_center, y_center, width, height, angle in degrees.
- **confidence** (float): Confidence of the detection.
- **label** (int): Label of the detection.
- **keypoints** (List\[[Keypoint](#keypoint)\]): Keypoints of the detection.

## ImgDetectionsExtended

ImgDetectionsExtended class for storing image detections with keypoints.

### Attributes

- **detections** (List\[[ImgDetectionExtended](#imgdetectionextended)\]): Image detections with keypoints.
- **masks** (np.ndarray): The segmentation masks of the image. All masks are stored in a single numpy array.

## Keypoint

Keypoint class for storing a keypoint.

### Attributes

- **x** (float): X coordinate of the keypoint, relative to the input height.
- **y** (float): Y coordinate of the keypoint, relative to the input width.
- **z** (Optional\[float\]): Z coordinate of the keypoint.
- **confidence** (Optional\[float\]): Confidence of the keypoint.

## Keypoints

Keypoints class for storing keypoints.

### Attributes

- **keypoints** (List\[[Keypoint](#keypoint)\]): List of Keypoint objects, each representing a keypoint.

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
