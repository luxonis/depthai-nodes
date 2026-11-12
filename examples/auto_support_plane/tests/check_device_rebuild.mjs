// Explicit hardware integration check: rebuilds the running showcase pipeline.
// Node 22+. Pass the RemoteConnection URL and an optional local evidence folder.
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {Remote,decodeCloud} from '../frontend/remote.js';

const url=process.argv[2];
if(!url) throw new Error('Pass ws://<OAK4-IP>:8765 explicitly; this test rebuilds its pipeline');
const directory=process.argv[3];
let remote;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function state(){const r=await remote.call('plane_control',{op:'state'});assert.ok(r.ok,r.error);return r.state;}
async function waitFor(predicate){
  const deadline=Date.now()+45000;let latest,error;
  while(Date.now()<deadline){
    try{latest=await state();if(latest.error)throw new Error(latest.error);if(predicate(latest))return latest;}
    catch(e){error=e;}
    await sleep(200);
  }
  throw new Error(`Rebuild did not finish: ${error?.message ?? JSON.stringify(latest)}`);
}
async function save(name,value){
  if(directory){await mkdir(directory,{recursive:true});await writeFile(`${directory}/${name}.json`,JSON.stringify(value,null,2));}
}
try{
  await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(new Error('Services not advertised')),10000);
    remote=new Remote(url,ready=>{if(ready){clearTimeout(timer);resolve();}});
  });
  const before=await waitFor(s=>s.phase==='holding' && s.last_result?.passed && s.held_plane);
  const originalSettings=structuredClone(before.settings);
  const restore=await remote.call('plane_control',{op:'restore',plane:before.held_plane});
  assert.ok(restore.ok,restore.error);
  const restored=await waitFor(s=>s.epoch>before.epoch && s.phase==='holding' && s.last_result?.passed);
  assert.deepEqual(restored.held_plane,before.held_plane);
  assert.equal(restored.calibration_outcome,null,'A valid initialPlane should not run a fit');
  assert.equal(restored.candidates.length,0);
  await save('restored',restored);
  console.log('PASS: native device reopened; exported plane accepted by fresh gate');

  const settings=structuredClone(originalSettings);settings.mode='on_start';settings.selection='automatic';
  const restart=await remote.call('plane_control',{op:'restart',settings});assert.ok(restart.ok,restart.error);
  const automatic=await waitFor(s=>s.epoch>restored.epoch && s.phase==='holding' && s.last_result?.passed);
  assert.equal(automatic.settings.mode,'on_start');assert.equal(automatic.settings.selection,'automatic');
  assert.equal(automatic.calibration_outcome,'succeeded');assert.equal(automatic.verification,null);
  assert.equal(automatic.waiting,false);
  await save('automatic-on-start',automatic);
  console.log('PASS: native settings rebuild; default automatic ON_START selection');

  const back=await remote.call('plane_control',{op:'restore',plane:automatic.held_plane,settings:originalSettings});
  assert.ok(back.ok,back.error);
  const final=await waitFor(s=>s.epoch>automatic.epoch && s.phase==='holding' && s.verification?.accepted);
  assert.equal(final.settings.mode,originalSettings.mode);
  assert.equal(final.settings.selection,originalSettings.selection);
  const cloud=decodeCloud(await remote.call('plane_cloud',{},true));
  assert.ok(cloud.metadata.count>0);assert.equal(cloud.metadata.frozen,false);
  await save('final-continuous',final);
  console.log(JSON.stringify({result:'PASS',phase:final.phase,settings:final.settings,
    verification:final.verification,displayed_points:cloud.metadata.count},null,2));
}finally{remote?.close();}
