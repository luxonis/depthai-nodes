"""Stamping and sending shared by the box associator's host nodes."""

import depthai as dai


def stamp_like(message: dai.Buffer, source: dai.Buffer) -> None:
    """Copy the timestamps and sequence number of ``source`` onto ``message``."""
    message.setTimestamp(source.getTimestamp())
    message.setTimestampDevice(source.getTimestampDevice())
    message.setSequenceNum(source.getSequenceNum())


def send_all(*sends: tuple[dai.Node.Output, dai.Buffer]) -> None:
    """Send each message on its output, stopping quietly once a queue is closed."""
    try:
        for output, message in sends:
            output.send(message)
    # DepthAI's stub declares QueueException through an undeclared nested
    # class, so pyright cannot tell it is an exception; at runtime it is one.
    except dai.MessageQueue.QueueException:  # pyright: ignore[reportGeneralTypeIssues]
        return
