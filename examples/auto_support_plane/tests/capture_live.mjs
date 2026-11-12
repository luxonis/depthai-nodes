// Capture explicit native service state/cloud evidence. Node 22+; read-only.
import {writeFile,mkdir} from 'node:fs/promises';
import {Remote,decodeCloud} from '../frontend/remote.js';

const url=process.argv[2] ?? 'ws://127.0.0.1:8765';
const directory=process.argv[3];
let remote;
try {
  await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(new Error('Services not advertised')),8000);
    remote=new Remote(url,ready=>{if(ready){clearTimeout(timer);resolve();}});
  });
  const response=await remote.call('plane_control',{op:'state'});
  if(!response.ok) throw new Error(response.error);
  const buffer=await remote.call('plane_cloud',{},true);
  const {metadata,points}=decodeCloud(buffer);
  const minimum=[Infinity,Infinity,Infinity],maximum=[-Infinity,-Infinity,-Infinity];
  for(let i=0;i<points.length;i++) {
    if(!Number.isFinite(points[i])) throw new Error('Non-finite cloud point');
    minimum[i%3]=Math.min(minimum[i%3],points[i]);
    maximum[i%3]=Math.max(maximum[i%3],points[i]);
  }
  const evidence={captured_at:new Date().toISOString(),url,state:response.state,cloud:{...metadata,minimum_mm:minimum,maximum_mm:maximum}};
  if(directory){
    await mkdir(directory,{recursive:true});
    await writeFile(`${directory}/service-state.json`,JSON.stringify(evidence,null,2));
    await writeFile(`${directory}/cloud.bin`,new Uint8Array(buffer));
  }
  console.log(JSON.stringify({phase:response.state.phase,ticket:response.state.ticket,
    candidates:response.state.candidates,verification:response.state.verification,cloud:evidence.cloud},null,2));
} finally {remote?.close();}
