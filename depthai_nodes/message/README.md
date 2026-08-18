# Message Types

Parser creators return native DepthAI messages. The parser-specific message types
are available under `dai.beta`:

- `dai.beta.Classifications`
- `dai.beta.Clusters` and `dai.beta.Cluster`
- `dai.beta.Keypoints`
- `dai.beta.Lines` and `dai.beta.Line`
- `dai.beta.Map2D`
- `dai.beta.Predictions` and `dai.beta.Prediction`

depthai-nodes retains only the package-specific messages below because DepthAI has
no native equivalent for them.

## Collection

`Collection` stores a list of messages or other items of the same runtime type.
Items can be added with `append(...)` or `extend(...)`.

## GatheredData

`GatheredData` stores a reference message and the messages gathered for that
reference. It inherits `Collection` and copies timestamp and sequence metadata from
the reference message.

## SnapData

`SnapData` represents a snap event uploaded to DepthAI Hub. It contains the snap
name, a `dai.FileGroup`, optional tags, and string metadata.
