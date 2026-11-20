const encoder = new TextEncoder();
const decoder = new TextDecoder();

export class CameraConnection {
  constructor(onConnection, onFrame, onVideoReset) {
    this.onConnection = onConnection;
    this.onFrame = onFrame;
    this.onVideoReset = onVideoReset;
    this.pending = new Map();
    this.nextCall = 1;
    this.service = null;
    this.channel = null;
    this.connect();
  }

  connect() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    this.socket = new WebSocket(`${protocol}//${location.host}/dai`, "foxglove.websocket.v1");
    this.socket.binaryType = "arraybuffer";
    this.socket.onmessage = ({ data }) => this.receive(data);
    this.socket.onclose = () => {
      this.service = null;
      this.channel = null;
      for (const { reject, timer } of this.pending.values()) {
        clearTimeout(timer);
        reject(new Error("Camera connection lost."));
      }
      this.pending.clear();
      this.onConnection(false);
      this.onVideoReset();
      setTimeout(() => this.connect(), 1000);
    };
  }

  receive(data) {
    if (typeof data === "string") {
      const message = JSON.parse(data);
      if (message.op === "advertiseServices") {
        const service = message.services.find(({ name }) => name === "detector");
        if (service) {
          this.service = service.id;
          this.onConnection(true);
        }
      } else if (message.op === "advertise") {
        const channel = message.channels.find(({ topic, schemaName }) =>
          topic === "RGB" && schemaName === "dai.proto.encoded_frame.EncodedFrame"
        );
        if (!channel || this.channel === channel.id) return;
        this.channel = channel.id;
        this.socket.send(JSON.stringify({
          op: "subscribe", subscriptions: [{ id: 1, channelId: channel.id }],
        }));
      } else if (message.op === "unadvertise" && message.channelIds.includes(this.channel)) {
        this.channel = null;
        this.onVideoReset();
      }
      return;
    }

    const view = new DataView(data);
    if (view.byteLength < 13) return;
    if (view.getUint8(0) === 1 && view.getUint32(1, true) === 1) {
      const timestamp = Number(view.getBigUint64(5, true) / 1000n);
      const payload = encodedFrameData(new Uint8Array(data, 13));
      if (payload) this.onFrame(payload, timestamp);
    } else if (view.getUint8(0) === 3) {
      const id = view.getUint32(5, true);
      const call = this.pending.get(id);
      if (!call) return;
      clearTimeout(call.timer);
      this.pending.delete(id);
      try {
        const offset = 13 + view.getUint32(9, true);
        const response = JSON.parse(decoder.decode(new Uint8Array(data, offset)));
        if (response.error) call.reject(new Error(response.error));
        else call.resolve(response);
      } catch (error) {
        call.reject(error);
      }
    }
  }

  call(action, config) {
    if (this.service === null || this.socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("Camera is disconnected."));
    }
    const id = this.nextCall++;
    const payload = encoder.encode(JSON.stringify({ action, config }));
    const bytes = new Uint8Array(17 + payload.length);
    const view = new DataView(bytes.buffer);
    view.setUint8(0, 2);
    view.setUint32(1, this.service, true);
    view.setUint32(5, id, true);
    view.setUint32(9, 4, true);
    bytes.set(encoder.encode("json"), 13);
    bytes.set(payload, 17);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error("Camera request timed out."));
      }, 3000);
      this.pending.set(id, { resolve, reject, timer });
      this.socket.send(bytes);
    });
  }
}

// Read the bytes field (16) from DepthAI's EncodedFrame protobuf.
function encodedFrameData(bytes) {
  let index = 0;
  function varint() {
    let value = 0;
    for (let shift = 0; shift < 35 && index < bytes.length; shift += 7) {
      const byte = bytes[index++];
      value += (byte & 127) * 2 ** shift;
      if (byte < 128) return value;
    }
    return NaN;
  }
  while (index < bytes.length) {
    const tag = varint();
    if (!tag) return null;
    switch (tag % 8) {
      case 0:
        // Skip numeric fields including 64-bit sequence values.
        while (index < bytes.length && bytes[index++] >= 128) {}
        break;
      case 1:
        index += 8;
        break;
      case 2: {
        const size = varint();
        if (size > bytes.length - index) return null;
        if (tag === 16 * 8 + 2) return bytes.subarray(index, index + size);
        index += size;
        break;
      }
      case 5:
        index += 4;
        break;
      default:
        return null;
    }
  }
  return null;
}

function h264Info(bytes) {
  let key = false;
  let codec;
  for (let i = 0; i + 3 < bytes.length; i++) {
    if (bytes[i] !== 0 || bytes[i + 1] !== 0 || bytes[i + 2] !== 1) continue;
    const type = bytes[i + 3] & 31;
    if (type === 5) key = true;
    if (type === 7 && i + 6 < bytes.length) {
      const profile = bytes.subarray(i + 4, i + 7);
      codec = "avc1." + Array.from(profile, (value) => value.toString(16).padStart(2, "0")).join("");
    }
  }
  return { key, codec };
}

export class CameraVideo {
  constructor(canvas, status) {
    this.canvas = canvas;
    this.status = status;
    this.context = canvas.getContext("2d");
    this.decoder = null;
    this.reset();
  }

  reset(message = "Waiting for video") {
    if (this.decoder && this.decoder.state !== "closed") this.decoder.close();
    this.decoder = null;
    this.status.hidden = false;
    this.status.textContent = "VideoDecoder" in window ? message : "Open this HTTPS page in Chrome to view H.264.";
  }

  push(data, timestamp) {
    if (!("VideoDecoder" in window)) return;
    const { key, codec } = h264Info(data);
    if (this.decoder?.decodeQueueSize > 12) this.reset();
    if (!this.decoder) {
      if (!key || !codec) return;
      this.decoder = new VideoDecoder({
        output: (frame) => {
          this.context.drawImage(frame, 0, 0, this.canvas.width, this.canvas.height);
          frame.close();
          this.status.hidden = true;
        },
        error: () => this.reset("Waiting for the next H.264 key frame"),
      });
      this.decoder.configure({ codec, optimizeForLatency: true });
    }
    this.decoder.decode(new EncodedVideoChunk({ type: key ? "key" : "delta", timestamp, data }));
  }
}
