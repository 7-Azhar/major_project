'use strict';
const $ = id => document.getElementById(id);
let state = null, options = null, serverId = null, sequence = -1, pending = -1;
let online = false, queued = 0, controlQueue = Promise.resolve(), upload = null, controlError = '';
let lastRecording = '', seeking = false;
const modes = [...document.querySelectorAll('[data-mode]')];
const clock = n => Math.floor(n / 60) + ':' + String(Math.floor(n % 60)).padStart(2, '0');

async function stationHeaders() {
  if (!document.body.dataset.stationToken) return {};
  // The embedded page may outlive a ground-station process restart.
  const response = await fetch('../api/session', {signal: AbortSignal.timeout(8000)});
  if (!response.ok) throw Error('Ground station session unavailable');
  const session = await response.json();
  return {'X-Station-Token': session.token};
}

async function jsonRequest(url, init = {}) {
  if (init.method === 'POST') init.headers = {...init.headers, ...await stationHeaders()};
  const response = await fetch(url, {...init, signal: AbortSignal.timeout(8000)});
  const result = await response.json();
  if (!response.ok) throw Error(result.error || (typeof result.detail === 'string' ? result.detail : '') || 'AURA could not complete the request.');
  return result;
}
function refreshAvailability() {
  const active = online && !!state;
  const ready = active && state.config.powered && !!state.output && queued === 0 && state.revision >= pending;
  const live = state?.config.source.startsWith('camera:');
  for (const id of ['source', 'enhancement', 'cameraIndex', 'connectCamera', 'outlines']) $(id).disabled = !active;
  $('videoFile').disabled = !active || !!upload;
  modes.forEach(button => button.disabled = !active);
  $('power').disabled = !active || queued > 0;
  $('play').disabled = !ready;
  $('restart').disabled = !ready || live;
  $('seek').disabled = !ready || live || !state.total_frames;
  $('disconnectCamera').disabled = !active || !live;
  $('saveReview').disabled = !ready;
  $('fullscreen').disabled = !ready || !document.fullscreenEnabled;
}
function control(change) {
  queued++;
  refreshAvailability();
  $('status').textContent = 'Applying…';
  const task = controlQueue.then(async () => {
    const result = await jsonRequest('api/control', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(change)});
    if (result.server_id && serverId !== result.server_id) {
      serverId = result.server_id;
      sequence = -1;
      pending = -1;
      options = null;
    }
    pending = Math.max(pending, result.revision);
    controlError = '';
    return result;
  });
  controlQueue = task.catch(error => {controlError = error.message; $('error').textContent = controlError;}).finally(() => {
    queued--;
    refreshAvailability();
  });
  return task;
}
const change = values => {void control(values).catch(() => {});};
function setSelected(id, value) {
  if (document.activeElement !== $(id)) $(id).value = value;
}
async function loadOptions() {
  options = await jsonRequest('api/options');
  $('source').replaceChildren(new Option('Choose an input…', ''), ...options.sources.map(source => new Option(source.split('/').pop(), source)));
  $('enhancement').replaceChildren(...Object.entries(options.enhancements).map(([key, label]) => new Option(label, key)));
  if (!options.sources.includes(lastRecording)) lastRecording = options.sources[0] || '';
}
function render(s) {
  state = s;
  const powered = s.config.powered !== false;
  const live = s.config.source.startsWith('camera:');
  const hasFrame = powered && !!s.output && !s.error;
  $('power').textContent = powered ? 'Power off' : 'Power on';
  $('power').setAttribute('aria-label', powered ? 'Power off pipeline' : 'Power on pipeline');
  $('power').classList.toggle('primary', !powered);
  $('error').textContent = controlError || s.error || '';
  $('status').textContent = s.status === 'playing' && live ? 'Live' : s.status[0].toUpperCase() + s.status.slice(1);
  $('emptyState').classList.toggle('hidden', hasFrame);
  $('feeds').classList.toggle('hidden', !hasFrame);
  if (!hasFrame) {
    $('emptyTitle').textContent = !powered ? 'Pipeline powered off' : s.error ? 'Input needs attention' : s.status === 'idle' ? 'Choose your input' : 'Preparing the view';
    $('emptyMessage').textContent = !powered ? 'Enhancement and detection are stopped. Power on to resume.' : (s.error || s.status === 'idle') ? 'Choose a recording or connect your camera using Input & settings.' : 'Your first frame will appear here.';
    for (const id of ['enhanceTime', 'detectTime', 'count', 'excluded']) $(id).textContent = '—';
    for (const id of ['raw', 'output']) $(id).removeAttribute('src');
    $('objects').textContent = powered ? 'No current frame' : 'Processing stopped';
    $('pipeline').textContent = powered ? 'Waiting for input' : 'Standby · selected settings are preserved';
    $('time').textContent = '— / —';
    $('device').textContent = 'Local processing';
  }
  if (s.config.source && ![...$('source').options].some(option => option.value === s.config.source)) {
    $('source').add(new Option(live ? 'Live ' + s.config.source : s.config.source.split('/').pop(), s.config.source));
  }
  if (!live && s.config.source) lastRecording = s.config.source;
  setSelected('source', s.config.source);
  setSelected('enhancement', s.config.enhancement);
  $('outlines').checked = s.config.outlines;
  $('play').textContent = live ? (s.config.playing ? 'Pause camera' : 'Resume camera') : (s.config.playing ? 'Pause' : s.status === 'ended' ? 'Replay' : 'Play');
  modes.forEach(button => button.setAttribute('aria-pressed', button.dataset.mode === s.config.detection));
  const exclude = s.config.detection.startsWith('exclude');
  $('warning').classList.toggle('hidden', !exclude);
  $('calibration').textContent = exclude ? 'FPV exclusion on · regions scale to this frame. Check their alignment using the outlines.' : 'FPV exclusion off · choose an Exclude mode whenever needed.';
  if (hasFrame) {
    $('output').src = 'data:image/jpeg;base64,' + s.output;
    if ($('compare').checked) $('raw').src = 'data:image/jpeg;base64,' + s.original;
    $('seek').max = Math.max(0, s.total_frames - 1);
    if (!seeking) $('seek').value = s.frame;
    $('time').textContent = live ? (s.config.playing ? 'LIVE' : 'CAMERA PAUSED') : clock(s.frame / s.source_fps) + ' / ' + clock(s.total_frames / s.source_fps);
    $('enhanceTime').textContent = s.enhancement_ms + ' ms';
    $('detectTime').textContent = s.config.detection === 'off' ? 'Off' : s.detection_ms + ' ms';
    $('count').textContent = s.detections.length;
    $('excluded').textContent = s.excluded_count;
    $('device').textContent = s.gpu + ' · ' + (s.hardware?.precision || '') + ' · ' + s.width + ' × ' + s.height + ' · Frame ' + s.frame;
    $('timingNote').textContent = s.inference_reused ? 'Same-frame predictions reused. Timings show the original processing cost, not camera-to-screen latency.' : 'Timings measure processing, not camera-to-screen latency.';
    $('pipeline').textContent = options.enhancements[s.config.enhancement] + ' → ' + (s.config.detection === 'off' ? 'Enhancement only' : 'YOLOv8m VisDrone');
    $('outputLabel').textContent = options.enhancements[s.config.enhancement].toUpperCase() + ' / ' + s.config.detection.toUpperCase();
    const counts = {};
    s.detections.forEach(detection => counts[detection.label] = (counts[detection.label] || 0) + 1);
    $('objects').textContent = Object.entries(counts).map(([name, count]) => name + ' × ' + count).join(' · ') || (s.config.detection === 'off' ? 'Detection is switched off.' : 'No detections above the selected threshold.');
  }
  refreshAvailability();
}
async function poll() {
  try {
    const query = serverId ? '?after=' + sequence + '&server_id=' + encodeURIComponent(serverId) : '';
    const response = await fetch('api/frame' + query, {signal: AbortSignal.timeout(8000)});
    if (!response.ok) throw Error('Pipeline unavailable');
    const reconnected = !online;
    online = true;
    $('connection').textContent = 'Connected · local';
    if (response.status !== 204) {
      const packet = await response.json();
      if (packet.server_id !== serverId) {
        serverId = packet.server_id;
        pending = -1;
        options = null;
        controlError = '';
      }
      sequence = packet.sequence;
      if (!options) await loadOptions();
      // Re-fetch the accepted frame after queued controls finish, even while paused.
      if (packet.revision >= pending && queued === 0) render(packet);
      else sequence = -1;
    } else if (reconnected && state) {
      render(state);
    }
    refreshAvailability();
  } catch (error) {
    online = false;
    sequence = -1;
    $('connection').textContent = 'Reconnecting…';
    $('status').textContent = 'Offline';
    $('error').textContent = 'Cannot reach AURA. Keep the launcher running; this page reconnects automatically.';
    $('feeds').classList.add('hidden');
    $('emptyState').classList.remove('hidden');
    $('emptyTitle').textContent = 'Connection interrupted';
    $('emptyMessage').textContent = 'Waiting for your local pipeline to return.';
    refreshAvailability();
  } finally {
    setTimeout(poll, online ? 80 : 1500);
  }
}
$('source').onchange = event => change({source: event.target.value, playing: false});
$('enhancement').onchange = event => change({enhancement: event.target.value});
$('outlines').onchange = event => change({outlines: event.target.checked});
$('compare').onchange = event => {
  $('feeds').classList.toggle('compare', event.target.checked);
  if (event.target.checked && state?.original) $('raw').src = 'data:image/jpeg;base64,' + state.original;
};
modes.forEach(button => button.onclick = () => change({detection: button.dataset.mode}));
$('power').onclick = () => change({powered: !state.config.powered});
$('play').onclick = () => change(state.status === 'ended' ? {seek: 0, playing: true} : {playing: !state.config.playing});
$('restart').onclick = () => change({seek: 0});
$('seek').oninput = () => {seeking = true;};
$('seek').onchange = event => {seeking = false; change({seek: Number(event.target.value)});};
$('seek').onblur = () => {seeking = false;};
$('fullscreen').onclick = async () => {
  try {if (document.fullscreenElement) await document.exitFullscreen(); else await $('viewer').requestFullscreen();}
  catch { $('reviewStatus').textContent = 'Full screen is unavailable in this browser window.'; }
};
$('exitFullscreen').onclick = () => {void document.exitFullscreen().catch(() => {});};
document.onkeydown = event => {
  if (event.code === 'Escape' && document.fullscreenElement) {void document.exitFullscreen().catch(() => {});}
  if (event.code === 'Space' && !event.repeat && !event.target.closest('button,input,select,textarea,summary,a,[contenteditable]') && !$('play').disabled) {
    event.preventDefault(); $('play').click();
  }
};
for (let index = 0; index < 10; index++) $('cameraIndex').add(new Option('Camera ' + index + (index === 0 ? ' (usually built-in)' : ''), index));
$('connectCamera').onclick = () => change({source: 'camera:' + $('cameraIndex').value, playing: true});
$('disconnectCamera').onclick = () => change({source: lastRecording || '', playing: false});
$('cancelUpload').onclick = () => upload?.abort();
$('videoFile').onchange = async () => {
  const file = $('videoFile').files[0];
  if (!file) return;
  if (file.size === 0 || file.size > 1024 ** 3) { $('uploadStatus').textContent = 'Choose a nonempty video up to 1 GB.'; $('videoFile').value = ''; return; }
  let auth;
  try { auth = await stationHeaders(); }
  catch (error) { $('uploadStatus').textContent = error.message; $('videoFile').value = ''; return; }
  const request = upload = new XMLHttpRequest();
  refreshAvailability();
  $('cancelUpload').classList.remove('hidden');
  $('uploadStatus').textContent = 'Copying locally…';
  request.open('POST', 'api/upload');
  for (const [name, value] of Object.entries(auth)) request.setRequestHeader(name, value);
  request.timeout = 5 * 60 * 1000;
  request.setRequestHeader('Content-Type', 'application/octet-stream');
  request.setRequestHeader('X-Filename', encodeURIComponent(file.name));
  request.upload.onprogress = event => {if (event.lengthComputable) $('uploadStatus').textContent = event.loaded === event.total ? 'Checking the video…' : 'Copying locally: ' + Math.round(event.loaded / event.total * 100) + '%';};
  const finish = () => {upload = null; $('videoFile').value = ''; $('cancelUpload').classList.add('hidden'); refreshAvailability();};
  request.onerror = () => { $('uploadStatus').textContent = 'Upload failed. Check the local server.'; finish(); };
  request.ontimeout = () => { $('uploadStatus').textContent = 'Upload timed out. Please try again.'; finish(); };
  request.onabort = () => { $('uploadStatus').textContent = 'Upload cancelled.'; finish(); };
  request.onload = async () => {
    try {
      const result = JSON.parse(request.responseText);
      if (request.status !== 200) throw Error(result.error || 'Upload failed');
      await loadOptions();
      await control({source: result.source, playing: false});
      $('uploadStatus').textContent = 'Selected: ' + file.name;
    } catch (error) { $('uploadStatus').textContent = error.message; }
    finally {finish();}
  };
  request.send(file);
};
$('saveReview').onclick = () => {
  // Use this exact displayed packet; the live worker may already have advanced.
  const packet = state;
  const documentCopy = document.implementation.createHTMLDocument('AURA frame ' + packet.frame);
  const style = documentCopy.createElement('style');
  style.textContent = 'body{font:15px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;color:#172324}section{display:flex;gap:20px;flex-wrap:wrap}figure{flex:1;min-width:260px;margin:0}img{width:100%}pre{background:#eef2f0;padding:20px;overflow:auto}figcaption{padding:10px 0}';
  documentCopy.head.append(style);
  const heading = documentCopy.createElement('h1'); heading.textContent = 'AURA · Review frame ' + packet.frame;
  const description = documentCopy.createElement('p');
  description.textContent = packet.config.source + ' · ' + options.enhancements[packet.config.enhancement] + ' · ' + packet.config.detection;
  const images = documentCopy.createElement('section');
  for (const [key, caption] of [['original', 'Original input'], ['output', 'Pipeline output']]) {
    const figure = documentCopy.createElement('figure'), image = documentCopy.createElement('img'), label = documentCopy.createElement('figcaption');
    image.src = 'data:image/jpeg;base64,' + packet[key]; image.alt = caption; label.textContent = caption; figure.append(image, label); images.append(figure);
  }
  const note = documentCopy.createElement('p');
  note.textContent = 'A review snapshot, not a labelled accuracy evaluation. Excluded detections and their reasons are recorded below. Coordinates refer to the original frame dimensions. Images are JPEG previews.';
  const metadata = {...packet}; delete metadata.original; delete metadata.output;
  const pre = documentCopy.createElement('pre'); pre.textContent = JSON.stringify(metadata, null, 2);
  documentCopy.body.append(heading, description, images, note, pre);
  const url = URL.createObjectURL(new Blob(['<!doctype html>\n' + documentCopy.documentElement.outerHTML], {type: 'text/html;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url;
  link.download = 'aura-frame-' + packet.frame + '-' + new Date().toISOString().replace(/[:.]/g, '-') + '.html';
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 60000);
  $('reviewStatus').textContent = 'Review download requested: original, output, settings, and detection details in one HTML file.';
};
poll();
