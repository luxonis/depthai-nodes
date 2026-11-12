import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeCloud, Remote} from '../frontend/remote.js';

test('binary cloud metadata may have unaligned length; XYZ remains exact',()=>{
  const metadata = new TextEncoder().encode(JSON.stringify({count:2,sequence:7,frozen:true}));
  const points = new Float32Array([0,0,1000,10,20,1500]);
  const buffer = new ArrayBuffer(4+metadata.length+points.byteLength);
  new DataView(buffer).setUint32(0,metadata.length,true);
  new Uint8Array(buffer,4,metadata.length).set(metadata);
  new Uint8Array(buffer,4+metadata.length).set(new Uint8Array(points.buffer));
  const result = decodeCloud(buffer);
  assert.equal(result.metadata.sequence,7);
  assert.deepEqual(result.points,points);
});

test('malformed and truncated cloud frames fail explicitly',()=>{
  assert.throws(()=>decodeCloud(new ArrayBuffer(0)));
  const buffer = new ArrayBuffer(8);
  new DataView(buffer).setUint32(0,999,true);
  assert.throws(()=>decodeCloud(buffer));
});

// Optional native service round trip against tests/serve_simulation.py.
test('native RemoteConnection services return state and packed cloud',{
  skip: !process.env.PLANE_WS_URL,
},async()=>{
  let remote;
  await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(new Error('No native service advertisement')),5000);
    remote = new Remote(process.env.PLANE_WS_URL,(ready)=>{if(ready){clearTimeout(timer);resolve();}});
  });
  try {
    const response=await remote.call('plane_control',{op:'state'});
    assert.equal(response.ok,true);
    assert.ok(response.state.settings);
    const decoded=decodeCloud(await remote.call('plane_cloud',{},true));
    assert.equal(decoded.points.length,decoded.metadata.count*3);
  } finally {remote.close();}
});
