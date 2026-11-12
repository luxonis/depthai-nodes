// The native Foxglove WebSocket service framing used by dai.RemoteConnection.
// Reference: depthai-core/3rdparty/foxglove/ws-protocol, ServiceResponse::write.
export class Remote {
  constructor(url, onStatus) {
    this.url = url;
    this.onStatus = onStatus;
    this.services = new Map();
    this.pending = new Map();
    this.nextCall = 1;
    this.stopped = false;
    this.connect();
  }

  connect() {
    const socket = this.socket = new WebSocket(this.url, 'foxglove.websocket.v1');
    socket.binaryType = 'arraybuffer';
    socket.onopen = () => this.onStatus(false, 'Discovering services…');
    socket.onerror = () => socket.close();
    socket.onmessage = ({ data }) => {
      try {
        if (typeof data === 'string') {
          const message = JSON.parse(data);
          if (message.op === 'advertiseServices') {
            for (const service of message.services) this.services.set(service.name, service);
            this.onStatus(this.services.has('plane_control') && this.services.has('plane_cloud'), 'Connected');
          } else if (message.op === 'unadvertiseServices') {
            for (const [name, service] of this.services) {
              if (message.serviceIds.includes(service.id)) this.services.delete(name);
            }
          } else if (message.op === 'serviceCallFailure') {
            this.finish(message.callId, new Error(message.message));
          }
        } else {
          const view = new DataView(data);
          if (view.byteLength < 13 || view.getUint8(0) !== 3) return;
          const callId = view.getUint32(5, true);
          const offset = 13 + view.getUint32(9, true);
          if (offset > data.byteLength) throw new Error('Invalid service response');
          this.finish(callId, null, data.slice(offset));
        }
      } catch (error) {
        this.onStatus(false, error.message);
      }
    };
    socket.onclose = () => {
      this.services.clear();
      for (const id of this.pending.keys()) this.finish(id, new Error('Connection closed'));
      this.onStatus(false, 'Reconnecting…');
      if (!this.stopped) this.reconnect = setTimeout(() => this.connect(), 1500);
    };
  }

  finish(id, error, payload) {
    const pending = this.pending.get(id);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pending.delete(id);
    if (error) pending.reject(error);
    else pending.resolve(payload);
  }

  async call(name, request = {}, binary = false) {
    const service = this.services.get(name);
    if (!service || this.socket.readyState !== WebSocket.OPEN) throw new Error('Device is disconnected');
    const encoding = new TextEncoder().encode(binary ? 'binary' : 'json');
    const data = binary ? new Uint8Array() : new TextEncoder().encode(JSON.stringify(request));
    const callId = this.nextCall++;
    const frame = new Uint8Array(13 + encoding.length + data.length);
    const view = new DataView(frame.buffer);
    view.setUint8(0, 2);
    view.setUint32(1, service.id, true);
    view.setUint32(5, callId, true);
    view.setUint32(9, encoding.length, true);
    frame.set(encoding, 13);
    frame.set(data, 13 + encoding.length);
    const payload = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.finish(callId, new Error(`${name} timed out`)), 5000);
      this.pending.set(callId, { resolve, reject, timer });
      try { this.socket.send(frame); } catch (error) { this.finish(callId, error); }
    });
    return binary ? payload : JSON.parse(new TextDecoder().decode(payload));
  }

  close() {
    this.stopped = true;
    clearTimeout(this.reconnect);
    this.socket.close();
  }
}

export function decodeCloud(buffer) {
  if (buffer.byteLength < 4) throw new Error('Missing cloud header');
  const length = new DataView(buffer).getUint32(0, true);
  if (length > buffer.byteLength - 4) throw new Error('Invalid cloud metadata');
  const metadata = JSON.parse(new TextDecoder().decode(buffer.slice(4, 4 + length)));
  const xyz = buffer.slice(4 + length);
  if (!Number.isInteger(metadata.count) || metadata.count < 0 || xyz.byteLength !== metadata.count * 12) {
    throw new Error('Invalid cloud point count');
  }
  return { metadata, points: new Float32Array(xyz) };
}
