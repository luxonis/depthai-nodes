import { Remote, decodeCloud } from './remote.js';
import { CloudView } from './cloud.js';

const $ = id => document.getElementById(id);
const integers = new Set(['failed_checks_before_refit','max_calibration_attempts','ransac_sample_points','ransac_max_iterations','max_plane_passes']);
const advanced = [
  ['distance_threshold_mm','RANSAC band (mm)',0.1,1],
  ['ransac_sample_points','RANSAC sample points',3,1],
  ['ransac_max_iterations','RANSAC iterations',1,1],
  ['max_plane_passes','Plane passes per cloud',1,1],
  ['min_inlier_ratio','Minimum inlier ratio',0,0.01,1],
  ['voxel_size_mm','Voxel size (mm)',0.1,1],
  ['max_gravity_angle_deg','Maximum tilt (°)',0,0.5,89.9],
  ['merge_distance_threshold_mm','Merge distance (mm)',0.1,1],
];
for (const [name,title,min,step,max] of advanced) {
  const label = document.createElement('label'); label.textContent = title;
  const input = document.createElement('input');
  Object.assign(input,{name,type:'number',min:String(min),step:integers.has(name) ? String(step) : 'any'});
  if (max !== undefined) input.max = max;
  label.append(input); $('fitting-fields').append(label);
}

let remote, viewer, state, connected = false, settingsEpoch = -1, candidateKey = '', busy = false;
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
function download(name, content) {
  const url = URL.createObjectURL(new Blob([content],{type:'application/json'}));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}

async function command(op, fields = {}) {
  if (busy) return;
  busy = true; updateButtons();
  try {
    const response = await remote.call('plane_control',{
      op,epoch:state?.epoch,ticket:state?.ticket,index:state?.preview_index ?? 0,...fields,
    });
    if (!response.ok) throw new Error(response.error);
    showError('');
    if (response.state) render(response.state);
    return response;
  } catch (error) { showError(error.message); }
  finally { busy = false; updateButtons(); }
}

function updateButtons() {
  const disabled = !connected || busy || !state;
  $('confirm').disabled = disabled || !state?.waiting;
  $('reject').disabled = disabled || !state?.waiting;
  $('export').disabled = disabled || !state?.held_plane || state?.phase === 'verifying restored plane';
  for (const id of ['search','import','forget']) $(id).disabled = disabled;
  $('settings').querySelector('button[type=submit]').disabled = disabled;
  for (const button of $('candidates').querySelectorAll('button')) button.disabled = disabled || !state?.waiting;
}

function fillSettings(settings) {
  for (const [name,value] of Object.entries({...settings,...settings.calibration})) {
    const input = $('settings').elements.namedItem(name);
    if (input) input.value = value ?? '';
  }
}

function readSettings() {
  const settings = structuredClone(state.settings);
  for (const [name,raw] of new FormData($('settings'))) {
    if (name === 'mode' || name === 'selection') { settings[name] = raw; continue; }
    const value = name === 'max_calibration_attempts' && raw.trim() === '' ? null : Number(raw);
    if (value !== null && (!Number.isFinite(value) || (integers.has(name) && !Number.isInteger(value)))) {
      throw new Error(`Invalid ${name}`);
    }
    if (name in settings.calibration) settings.calibration[name] = value;
    else settings[name] = value;
  }
  return settings;
}

function render(next) {
  state = next;
  if (settingsEpoch !== state.epoch) { fillSettings(state.settings); settingsEpoch = state.epoch; }
  $('phase').textContent = state.phase;
  $('selection-title').textContent = state.waiting ? 'Choose your support plane' : state.held_plane ? 'A plane to build on' : 'Find your support surface';
  $('selection-help').textContent = state.waiting
    ? 'Preview each candidate, then confirm the surface you want. The fitted cloud is frozen for comparison.'
    : state.held_plane
      ? (state.settings.mode === 'continuous' ? 'The live cloud is checked against your plane. Move the camera to try a refit.' : 'On-start mode holds the plane without periodic checks. Search again to recalibrate.')
      : 'Keep the camera still and point it toward a table or floor. Searching waits for fresh accelerometer data.';
  const key = `${state.epoch}:${state.ticket}:${state.waiting}:${JSON.stringify(state.candidates)}`;
  if (key !== candidateKey) {
    candidateKey = key; $('candidates').replaceChildren();
    state.candidates.forEach((candidate,index) => {
      const button = document.createElement('button'); button.className = 'candidate';
      button.setAttribute('aria-pressed','false');
      const row = document.createElement('div'); row.className = 'row';
      const name = document.createElement('span'); name.className = 'name'; name.textContent = `Plane ${String(index+1).padStart(2,'0')}`;
      const ratio = document.createElement('span'); ratio.className = 'ratio'; ratio.textContent = `${(candidate.inlier_ratio*100).toFixed(1)}% inliers`;
      row.append(name,ratio);
      const numbers = document.createElement('div'); numbers.className = 'numbers';
      numbers.textContent = `${candidate.inlier_count.toLocaleString()} inliers · offset ${candidate.plane.offset_mm.toFixed(0)} mm`;
      button.append(row,numbers);
      button.addEventListener('click',()=>command('preview',{index}));
      $('candidates').append(button);
    });
  }
  [...$('candidates').children].forEach((button,index) => {
    const active = state.waiting && index === state.preview_index;
    button.classList.toggle('active',active); button.setAttribute('aria-pressed',String(active));
  });
  const verification = state.verification;
  $('coverage').textContent = verification?.on_plane_ratio == null ? '—' : `${(verification.on_plane_ratio*100).toFixed(1)}%`;
  $('tilt').textContent = verification ? `${verification.gravity_angle_deg.toFixed(1)}°` : '—';
  $('check-status').textContent = verification ? (verification.accepted ? '✓ Last check passed' : '↻ Last check rejected') : 'No verification yet';
  $('result-status').textContent = state.last_result ? `passed = ${state.last_result.passed}` : 'No result yet';
  $('fit-outcome').textContent = state.calibration_outcome ? `Last calibration attempt: ${state.calibration_outcome}` : 'No calibration attempt finished';
  $('saved-status').textContent = state.saved ? 'Stored on device' : 'No saved plane';
  $('plane-equation').textContent = state.held_plane ? JSON.stringify(state.held_plane,null,2) : 'No confirmed plane';
  const warning = state.imu_age_sec == null ? 'Waiting for accelerometer data.' : state.imu_age_sec > 0.5 ? 'Accelerometer is stale. Calibration and verification wait for fresh data.' : state.cloud_age_sec == null || state.cloud_age_sec > 2 ? 'Waiting for a fresh organized depth cloud.' : '';
  $('sensor-warning').textContent = warning; $('sensor-warning').hidden = !warning;
  if (state.error) showError(state.error);
  $('events').replaceChildren(...state.events.slice().reverse().map(event => {
    const li = document.createElement('li'), time = document.createElement('time');
    time.textContent = event.at; li.append(time,document.createTextNode(event.text)); return li;
  }));
  updateButtons();
}

async function pollState() {
  if (connected) {
    try {
      const response = await remote.call('plane_control',{op:'state'});
      if (!response.ok) throw new Error(response.error);
      render(response.state);
    } catch (error) { showError(error.message); }
  }
  setTimeout(pollState,300);
}

async function pollCloud() {
  if (connected && viewer) {
    try {
      const {metadata,points} = decodeCloud(await remote.call('plane_cloud',{},true));
      viewer.setCloud(points,metadata);
      $('empty').hidden = metadata.count > 0;
      $('frame-kind').textContent = metadata.frozen ? 'CANDIDATE SNAPSHOT' : metadata.count ? 'LIVE CLOUD' : 'WAITING FOR CLOUD';
      $('frame-kind').classList.toggle('frozen',metadata.frozen);
      $('frame-number').textContent = `Frame ${metadata.sequence ?? '—'} · optical frame · mm`;
      $('point-count').textContent = `${metadata.count.toLocaleString()} displayed points`;
      $('legend-label').textContent = metadata.frozen ? 'Candidate plane' : 'Confirmed plane';
      document.querySelector('.legend.plane').style.background = metadata.frozen ? '#ffb64c' : '#6be0ba';
      $('cloud-caption').textContent = metadata.frozen
        ? (metadata.count ? `Fitted cloud frozen · points within ${metadata.band_mm} mm of this candidate are amber.` : 'Waiting for the exact fitted frame. Use “None of these · retry” if it is unavailable.')
        : `Live cloud · points within ${metadata.band_mm} mm of the held plane are green. Display stride 2; fitting uses all points.`;
    } catch (error) { showError(error.message); }
  }
  setTimeout(pollCloud,120);
}

$('confirm').onclick = () => command('confirm');
$('reject').onclick = () => command('reject');
$('search').onclick = () => command('restart');
$('forget').onclick = () => command('forget');
$('export').onclick = async () => {
  const response = await command('export');
  if (response) download('support_plane.json',JSON.stringify(response.plane,null,2));
};
$('import').onclick = () => $('plane-file').click();
$('plane-file').onchange = async event => {
  const file = event.target.files[0]; if (!file) return;
  try {
    if (file.size > 8192) throw new Error('Expected a small SupportPlane JSON file');
    await command('restore',{plane:JSON.parse(await file.text())});
  } catch (error) { showError(error.message); }
  event.target.value = '';
};
$('settings').onsubmit = event => {
  event.preventDefault();
  try { command('restart',{settings:readSettings()}); } catch (error) { showError(error.message); }
};
$('reset-view').onclick = () => viewer?.reset();
$('point-size').oninput = event => { if (viewer) { viewer.pointSize=Number(event.target.value); viewer.draw(); } };
$('plane-only').onchange = event => { if (viewer) { viewer.planeOnly=event.target.checked; viewer.draw(); } };
try { viewer = new CloudView($('cloud')); } catch (error) { showError(error.message); }

try {
  const configuration = await fetch('./connection.json',{cache:'no-store'}).then(response => {
    if (!response.ok) throw new Error('Run the Python app to serve the viewer and connection settings');
    return response.json();
  });
  const url = new URL(location.href); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  // oakapp-base proxies secure /dai to the native WebSocket server. Direct
  // local HTTP uses its separate WS port; do not attempt TLS on native 8765.
  if (location.protocol === 'https:') url.pathname = '/dai';
  else { url.port = configuration.websocket_port; url.pathname = '/'; }
  url.search = ''; url.hash = '';
  remote = new Remote(url.toString(),(ready,label) => {
    connected=ready; $('connection').textContent=configuration.simulation && ready ? 'Connected · synthetic scene' : label; $('connection-dot').classList.toggle('connected',ready);
    if (!ready) $('frame-kind').textContent = 'DISCONNECTED · LAST FRAME';
    updateButtons();
  });
  $('simulation').hidden = !configuration.simulation;
  for (const button of document.querySelectorAll('[data-demo]')) button.onclick = async () => {
    try { await remote.call('plane_demo',{op:button.dataset.demo}); } catch (error) { showError(error.message); }
  };
  pollState(); pollCloud();
} catch (error) { showError(error.message); }
window.addEventListener('pagehide',()=>remote?.close());
