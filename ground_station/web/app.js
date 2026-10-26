import {MissionMap,distance} from './map.js';
const $=id=>document.getElementById(id);
const fmt=(n,d=1)=>Number.isFinite(n)?n.toFixed(d):'—';
let token='',state=null,draft=[],dirty=false,selected=-1,history=[],proposal=null,saved=[],lastRevision=-1,lastMissionStatus='',online=false,refreshing=false,lastEventRefresh=0;
let toastTimer,settingsInitialised=false,serverId=null,events=null;
function toast(message,error=false){$('toast').textContent=message;$('toast').classList.toggle('error',error);$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,6500)}
async function request(path,body){
 const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Station-Token':token},body:JSON.stringify(body)};
 let response=await fetch(path,{...options,signal:AbortSignal.timeout(10000)});
 if(response.status===403&&body!==undefined){
  const session=await fetch('/api/session',{signal:AbortSignal.timeout(5000)});
  if(!session.ok)throw Error('Station session unavailable');
  token=(await session.json()).token;options.headers['X-Station-Token']=token;
  response=await fetch(path,{...options,signal:AbortSignal.timeout(10000)});
 }
 const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:data.error||JSON.stringify(data.detail));return data;
}
const safe=fn=>async(...args)=>{try{await fn(...args)}catch(e){toast(e.message,true)}};
function pushHistory(){history.push(JSON.stringify(draft));history=history.slice(-30)}
function canEdit(){if(state?.mission.status==='running'){toast('Pause the mission before changing its route',true);return false}return true}
function defaults(p){return {...p,altitude_m:Number($('default-alt').value),hold_s:Number($('default-hold').value)}}
function changeDraft(points,{remember=true,name}={}){if(!canEdit())return;if(remember)pushHistory();draft=structuredClone(points);dirty=true;if(name)$('mission-name').value=name;updateRows();map.points=draft;map.schedule()}
const map=new MissionMap($('map'),{
  onAdd:p=>{if(canEdit())changeDraft([...draft,defaults(p)])},
  onMove:(i,p)=>{draft[i]={...draft[i],...p};dirty=true;updateRows();map.points=draft;map.schedule()},
  onSelect:i=>{selected=i;map.selected=i;updateRows();map.schedule()},
  onHistory:pushHistory,
  onCursor:p=>$('map-cursor').textContent=fmt(p.latitude,6)+' / '+fmt(p.longitude,6)
});
function selectDock(name){document.querySelectorAll('[data-dock]').forEach(b=>b.classList.toggle('active',b.dataset.dock===name));document.querySelectorAll('.dock-pane').forEach(p=>p.classList.toggle('active',p.id==='dock-'+name))}
document.querySelectorAll('[data-dock]').forEach(b=>b.addEventListener('click',()=>selectDock(b.dataset.dock)));
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{
 document.querySelectorAll('[data-view]').forEach(n=>n.classList.toggle('active',n===b));
 const view=b.dataset.view;
 if(view==='simulation'){selectDock('simulation');$('dock-simulation').scrollIntoView({behavior:'smooth',block:'center'})}
 else if(view==='ai'){$('copilot-text').focus();$('copilot-form').scrollIntoView({behavior:'smooth',block:'center'})}
 else if(view==='flight'){$('horizon').scrollIntoView({behavior:'smooth',block:'center'})}
 else $('map').scrollIntoView({behavior:'smooth',block:'center'});
}));
function updateRows(){
 $('wp-count').textContent=draft.length;$('empty-route').hidden=!!draft.length;$('waypoint-rows').replaceChildren();
 draft.forEach((p,i)=>{
  const tr=document.createElement('tr');tr.classList.toggle('selected',i===selected);
  const td=text=>{const d=document.createElement('td');d.textContent=text;tr.append(d);return d};
  td(i+1);td('WAYPOINT').className='command-cell';
  ['latitude','longitude','altitude_m','hold_s'].forEach(field=>{
   const d=td(''),input=document.createElement('input');input.type='number';input.dataset.field=field;input.value=field.includes('itude')&&field!=='altitude_m'?p[field].toFixed(7):p[field];input.step=field==='altitude_m'||field==='hold_s'?'1':'.000001';input.setAttribute('aria-label',field.replaceAll('_',' ')+' waypoint '+(i+1));input.disabled=state?.mission.status==='running';input.addEventListener('change',()=>{if(!canEdit())return;pushHistory();draft[i][field]=Number(input.value);dirty=true;map.schedule();updateRows()});d.append(input);
  });
  td(i?fmt(distance(draft[i-1],p),1):'0.0');
  const order=td('');[-1,1].forEach(delta=>{const b=document.createElement('button');b.textContent=delta<0?'↑':'↓';b.setAttribute('aria-label',(delta<0?'Move up':'Move down')+' waypoint '+(i+1));b.disabled=i+delta<0||i+delta>=draft.length||state?.mission.status==='running';b.addEventListener('click',()=>{const points=structuredClone(draft);[points[i],points[i+delta]]=[points[i+delta],points[i]];changeDraft(points)});order.append(b)});
  const del=td(''),b=document.createElement('button');b.textContent='×';b.setAttribute('aria-label','Delete waypoint '+(i+1));b.disabled=state?.mission.status==='running';b.addEventListener('click',()=>changeDraft(draft.filter((_,j)=>j!==i)));del.append(b);$('waypoint-rows').append(tr);
 });
 const length=draft.slice(1).reduce((sum,p,i)=>sum+distance(draft[i],p),0);
 const speed=state?.limits.speed_mps||.5,seconds=length/speed+draft.reduce((sum,p)=>sum+p.hold_s,0);
 $('route-length').textContent=fmt(length)+' m planned';$('route-estimate').textContent=fmt(length)+' m · ~'+Math.ceil(seconds/60)+' min at '+speed+' m/s';
 $('draft-status').textContent=dirty?'Edited draft · validate before execution':draft.length?'Validated route · ready for review':'Create a route to begin';
}
function planBody(){return {name:$('mission-name').value.trim()||'Untitled search',points:draft}}
async function applyRoute(){const body=structuredClone(planBody()),result=await request('/api/mission',body);lastRevision=result.revision;dirty=JSON.stringify(body)!==JSON.stringify(planBody());updateRows();if(dirty)throw Error('Draft changed during validation. Review and validate again before starting.');return result}
$('apply-route').addEventListener('click',safe(async()=>{await applyRoute();toast('Route validated and applied')}));
$('map-add').addEventListener('click',()=>{map.addMode=true;$('map').classList.add('add-mode');$('map-add').classList.add('selected');$('map-pan').classList.remove('selected')});
$('map-pan').addEventListener('click',()=>{map.addMode=false;$('map').classList.remove('add-mode');$('map-pan').classList.add('selected');$('map-add').classList.remove('selected')});
$('map-fit').addEventListener('click',()=>map.fit());
$('zoom-in').addEventListener('click',()=>map.changeZoom(1));$('zoom-out').addEventListener('click',()=>map.changeZoom(-1));
$('map-centre').addEventListener('click',()=>{if(map.aircraft){map.centre={...map.aircraft};map.schedule()}});
$('map-layer').addEventListener('change',()=>{map.layer=$('map-layer').value;map.failures=0;map.schedule();if(map.layer!=='offline')toast('Online basemap enabled; provider receives the requested map area')});
$('route-add-current').addEventListener('click',()=>{if(!state||(state.telemetry.ages_s.position??Infinity)>2.5)return toast('Fresh position required',true);changeDraft([...draft,defaults({latitude:state.telemetry.latitude,longitude:state.telemetry.longitude})])});
$('route-clear').addEventListener('click',()=>changeDraft([]));
$('route-undo').addEventListener('click',()=>{if(!canEdit()||!history.length)return;changeDraft(JSON.parse(history.pop()),{remember:false})});
async function missionAction(action){if(!online||!state)throw Error('Station is reconnecting');if(action==='resume'&&dirty)throw Error('The draft has unapplied changes. Validate and start the revised route, or undo them before resuming.');if(action==='start'&&(dirty||!state.mission.points.length))await applyRoute();await request('/api/mission/action',{action});toast('Mission '+action+' requested');refreshEvents()}
['start','pause','resume','cancel'].forEach(action=>$('mission-'+action).addEventListener('click',safe(()=>missionAction(action))));
$('save-mission').addEventListener('click',safe(async()=>{await request('/api/missions/save',planBody());toast('Mission saved locally')}));
$('load-mission').addEventListener('click',safe(async()=>{saved=await request('/api/missions');if(!saved.length)return toast('No saved missions yet');const select=$('saved-missions');select.replaceChildren();const placeholder=document.createElement('option');placeholder.value='';placeholder.textContent='Choose a saved mission…';select.append(placeholder);saved.forEach(m=>{const option=document.createElement('option');option.value=m.id;option.textContent=m.name+' · '+m.points.length+' points';select.append(option)});select.hidden=false}));
$('saved-missions').addEventListener('change',()=>{const m=saved.find(m=>m.id===Number($('saved-missions').value));if(m){changeDraft(m.points,{name:m.name});map.fit()}});
function download(name,text,type='application/json'){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('export-route').addEventListener('click',()=>download('aero-mission.json',JSON.stringify({version:1,...planBody()},null,2)));
$('export-wpl').addEventListener('click',()=>{if(!draft.length||!state)return toast('Create a route first',true);const h=state.home;const rows=[['0','1','0','16','0','0','0','0',h.latitude,h.longitude,'0','1'],...draft.map((p,i)=>[i+1,0,3,16,p.hold_s,0,0,0,p.latitude,p.longitude,p.altitude_m,1])];download('aero-mission.waypoints','QGC WPL 110\n'+rows.map(r=>r.join('\t')).join('\n')+'\n','text/plain')});
$('import-route').addEventListener('click',()=>$('file-input').click());
$('file-input').addEventListener('change',safe(async()=>{
 const file=$('file-input').files[0];if(!file)return;if(file.size>200000)throw Error('Mission file must be under 200 KB');
 const text=await file.text();let points,name=file.name;
 if(text.trim().startsWith('QGC WPL 110')){const lines=text.trim().split(/\r?\n/).slice(1);points=[];for(const line of lines){const f=line.trim().split(/\s+/);if(f.length!==12)throw Error('Invalid WPL row');if(Number(f[0])===0&&Number(f[2])===0)continue;if(Number(f[2])!==3||Number(f[3])!==16)throw Error('Only relative-altitude WAYPOINT rows are supported; other commands are not silently ignored');points.push({latitude:Number(f[8]),longitude:Number(f[9]),altitude_m:Number(f[10]),hold_s:Number(f[4])})}}
 else{const obj=JSON.parse(text);if(obj.type==='FeatureCollection'){const line=obj.features.find(f=>f.geometry?.type==='LineString');if(!line)throw Error('GeoJSON needs a LineString');points=line.geometry.coordinates.map(c=>defaults({latitude:c[1],longitude:c[0]}))}else{points=obj.points;name=obj.name||name}}
 if(!Array.isArray(points)||points.length<1||points.length>100||points.some(p=>!['latitude','longitude','altitude_m','hold_s'].every(k=>Number.isFinite(p[k]))))throw Error('Invalid waypoint data');
 changeDraft(points,{name});map.fit();toast('Imported as a draft; validate before starting');$('file-input').value='';
}));
$('grid-form').addEventListener('submit',safe(async e=>{e.preventDefault();if(!state)throw Error('Station not ready');const t=state.telemetry;if((t.ages_s.position??Infinity)>2.5)throw Error('Fresh GPS required');const geo=await request('/api/search-grid',{latitude:t.latitude,longitude:t.longitude,width_m:Number($('grid-width').value),height_m:Number($('grid-height').value),spacing_m:Number($('grid-spacing').value)});changeDraft(geo.features[0].geometry.coordinates.map(c=>defaults({latitude:c[1],longitude:c[0]})),{name:'Lawnmower search'});map.fit();toast('Search route drafted; validate against the flight envelope')}));
async function propose(text){proposal=await request('/api/copilot/plan',{text});$('proposal').hidden=false;$('proposal-text').textContent=proposal.explanation+' '+proposal.points.length+' waypoints.';toast('Mission proposal generated')}
$('copilot-form').addEventListener('submit',safe(async e=>{e.preventDefault();await propose($('copilot-text').value)}));
document.querySelectorAll('[data-intent]').forEach(b=>b.addEventListener('click',()=>{$('copilot-text').value=b.dataset.intent;$('copilot-text').focus()}));
$('proposal-use').addEventListener('click',()=>{changeDraft(proposal.points,{name:proposal.name});map.fit();$('proposal').hidden=true});
$('return-home').addEventListener('click',safe(()=>propose('return home')));
$('ai-policy').addEventListener('change',safe(async()=>{await request('/api/ai/policy',{policy:$('ai-policy').value});toast('Vision response policy updated')}));
async function send(text){const result=await request('/api/command',{text,duration_s:.5});toast((result.mode==='demo'?'Simulated ':'Requested ')+result.requested+' · 0.5 s');refreshEvents()}
document.querySelectorAll('[data-command]').forEach(b=>b.addEventListener('click',safe(()=>send(b.dataset.command))));
$('command-form').addEventListener('submit',safe(async e=>{e.preventDefault();await send($('command').value)}));
$('accept-recommendation').addEventListener('click',safe(()=>send('hold')));
$('lease').addEventListener('change',safe(async()=>{try{await request('/api/lease',{enabled:$('lease').checked})}catch(e){$('lease').checked=false;throw e}}));
$('release').addEventListener('click',safe(async()=>{$('lease').checked=false;await request('/api/release');toast('Laptop control released')}));
setInterval(async()=>{if(online&&$('lease').checked&&!document.hidden){try{await request('/api/lease',{enabled:true})}catch(e){$('lease').checked=false;toast(e.message,true)}}},1000);
document.addEventListener('visibilitychange',()=>{if(document.hidden){$('lease').checked=false;request('/api/lease',{enabled:false}).catch(()=>{})}});
$('mark-person').addEventListener('click',()=>{selectDock('notebook');$('note').value='Possible rescue lead — verify the scene; saved coordinates identify aircraft';$('note').focus();$('note').scrollIntoView({behavior:'smooth',block:'center'})});
$('incident-form').addEventListener('submit',safe(async e=>{e.preventDefault();const result=await request('/api/incidents',{note:$('note').value});$('note').value='';toast(result.snapshot?'Note and displayed frame saved':'Note saved; no image was available');refreshEvents()}));
$('record-start').addEventListener('click',safe(async()=>{await request('/api/recording/start');toast('Recording started')}));
$('record-stop').addEventListener('click',safe(async()=>{await request('/api/recording/stop');toast('Recording saved');refreshEvents()}));
$('camera-expand').addEventListener('click',()=>$('video').closest('.camera-card').classList.toggle('full-camera'));
$('sim-apply').addEventListener('click',safe(async()=>{await request('/api/simulation',{person_visible:$('sim-person').checked,gps_loss:$('sim-gps').checked,video_loss:$('sim-video').checked,battery_percent:Number($('sim-battery').value),time_scale:Number($('sim-rate').value)});toast('Simulation scenario applied')}));
$('system-open').addEventListener('click',safe(async()=>{const settings=await request('/api/system');$('system-content').replaceChildren();for(const [key,value] of Object.entries(settings)){const p=document.createElement('p');p.textContent=key.replaceAll('_',' ')+': '+(typeof value==='object'?JSON.stringify(value):String(value));$('system-content').append(p)}$('system-dialog').showModal()}));
$('system-close').addEventListener('click',()=>$('system-dialog').close());
const Speech=window.SpeechRecognition||window.webkitSpeechRecognition;$('voice').disabled=!Speech;
if(Speech)$('voice').addEventListener('click',()=>{const recognition=new Speech();recognition.lang='en-IN';recognition.onresult=e=>{$('command').value=e.results[0][0].transcript;toast('Review dictated command, then Send')};recognition.onerror=e=>toast('Speech unavailable: '+e.error,true);recognition.start()});
function drawHorizon(t,demo){const c=$('horizon'),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);if(!demo&&(t.ages_s.attitude??Infinity)>2.5){ctx.fillStyle='#8b9ba5';ctx.font='12px monospace';ctx.fillText('ATTITUDE UNAVAILABLE',w/2-80,h/2);return}ctx.save();ctx.translate(w/2,h/2);ctx.rotate(-(t.roll_deg||0)*Math.PI/180);const shift=(t.pitch_deg||0)*2;ctx.fillStyle='#2e687f';ctx.fillRect(-w,-h*2,w*2,h*2+shift);ctx.fillStyle='#665438';ctx.fillRect(-w,shift,w*2,h*2);ctx.strokeStyle='#d7e3e8';ctx.beginPath();ctx.moveTo(-w,shift);ctx.lineTo(w,shift);ctx.stroke();ctx.font='9px monospace';ctx.fillStyle='#e2e5dd';[-20,-10,10,20].forEach(angle=>{const y=shift-angle*2;ctx.beginPath();ctx.moveTo(-25,y);ctx.lineTo(25,y);ctx.stroke();ctx.fillText(angle,30,y+3)});ctx.restore();ctx.strokeStyle='#e3ef76';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(w/2-40,h/2);ctx.lineTo(w/2-10,h/2);ctx.lineTo(w/2,h/2+6);ctx.lineTo(w/2+10,h/2);ctx.lineTo(w/2+40,h/2);ctx.stroke();ctx.fillStyle='#152026dd';ctx.fillRect(w/2-30,0,60,20);ctx.fillStyle='#e7f1e7';ctx.font='10px monospace';ctx.fillText(fmt(t.heading_deg,0)+'°',w/2-15,14)}
function render(s){
 document.querySelectorAll('[data-connection-disabled]').forEach(el=>{el.disabled=false;el.removeAttribute('data-connection-disabled')});
 if(serverId!==s.server_id){serverId=s.server_id;lastRevision=-1;lastMissionStatus='';settingsInitialised=false;$('lease').checked=false;}
 state=s;online=true;const t=s.telemetry,v=s.video,m=s.mission,demo=s.mode==='demo',posFresh=(t.ages_s.position??Infinity)<2.5,batFresh=(t.ages_s.battery??Infinity)<2.5;
 if(!settingsInitialised){settingsInitialised=true;$('ai-policy').value=s.intelligence.policy;if(s.simulation){$('sim-person').checked=s.simulation.person_visible;$('sim-gps').checked=s.simulation.gps_loss;$('sim-video').checked=s.simulation.video_loss;$('sim-battery').value=s.simulation.battery_percent;$('sim-rate').value=s.simulation.time_scale}}
 $('connection').textContent='Station online';$('mode').textContent=demo?'SIMULATION':'LIVE';$('connection-dot').style.background='var(--lime)';
 $('session-label').textContent=demo?'SIMULATED AIRCRAFT · NO RF OUTPUT':'LIVE AIRCRAFT · GUIDED SUPERVISION';
 $('session-message').textContent=demo?(v.backend==='aura'?'SIMULATED telemetry / real AURA video · no aircraft connection':(v.simulated?'Generated video / telemetry · hardware integration pending':'SIMULATED telemetry / real captured video · no aircraft connection')):s.mission_blockers.length?'Observation: '+s.mission_blockers[0]:'Flight prerequisites met · pilot retains takeover';
 $('altitude').textContent=posFresh?fmt(t.altitude_m):'—';$('speed').textContent=posFresh?fmt(t.speed_mps):'—';
 $('battery').textContent=batFresh&&t.battery_percent!==null?t.battery_percent+'%':'—';$('voltage').textContent=(batFresh?fmt(t.voltage_v):'—')+' V';
 $('sats').textContent=(t.ages_s.gps??Infinity)<2.5?t.satellites??'—':'—';$('gps').textContent=t.fix_type>=3?'3D fix':'No 3D fix';
 $('flight-mode').textContent=t.mode;$('armed').textContent=t.armed?'ARMED':'DISARMED';$('temperature').textContent=(demo||(t.ages_s.temperature??Infinity)<2.5)?fmt(t.temperature_c,0)+'°C barometer':'Temperature —';$('telemetry-status').textContent=t.status;
 const frameFresh=v.age_s!==null&&v.age_s<1;$('fps').textContent=(frameFresh?fmt(v.fps):'—')+' FPS';$('frame-age').textContent=fmt(v.age_s===null?null:v.age_s*1000,0)+' ms age';$('encode').textContent=fmt(v.processing_ms,0)+' ms encode';
 $('video').style.visibility=v.backend==='aura'&&(!v.connected||!v.aura.config.powered)?'hidden':'visible';
 $('video-source').textContent=v.backend==='aura'?'AURA / '+(v.live_ready?'CAMERA':'VIDEO'):v.simulated?'SIMULATED':'USB / FILE';$('video-badge').textContent=!frameFresh?'PAUSED / STALE':v.backend==='aura'?'AURA VIDEO':v.simulated?'SIMULATED VIDEO':'CAPTURE ACTIVE';
 $('action').textContent=m.status==='paused'?'Paused · '+m.reason:s.action;$('blockers').textContent=s.control_blockers.join(' · ');
 document.querySelectorAll('[data-command]').forEach(b=>b.disabled=!demo&&s.control_blockers.length>0);
 if(!s.lease_active&&!demo)$('lease').checked=false;
 $('mission-state').textContent=m.status.toUpperCase();$('mission-progress').textContent=Math.min(m.index,m.points.length)+' / '+m.points.length+' waypoints';$('mission-percent').textContent=m.progress_percent+'%';$('mission-progress-bar').style.width=m.progress_percent+'%';$('mission-reason').textContent=m.reason;$('mission-blockers').textContent=s.mission_blockers.join(' · ');
 $('mission-start').disabled=m.status==='running'||s.mission_blockers.length>0||!draft.length;$('mission-pause').disabled=m.status!=='running';$('mission-resume').disabled=m.status!=='paused'||s.mission_blockers.length>0||dirty;$('mission-cancel').disabled=!['running','paused'].includes(m.status);
 map.editable=m.status!=='running';
 if(m.status!==lastMissionStatus){lastMissionStatus=m.status;updateRows()}
 if(m.revision!==lastRevision){lastRevision=m.revision;if(!dirty){draft=structuredClone(m.points);if(m.points.length)$('mission-name').value=m.name;updateRows();map.points=draft}}
 $('ai-decision').textContent=s.intelligence.decision;$('track-count').textContent=s.intelligence.tracks.length+' image tracks · '+s.intelligence.confirmed_count+' persistent candidates';
 $('ai-status').textContent=v.ai_status;$('detections').textContent=v.detections.map(d=>d.label+' '+(d.confidence===null?d.detector:Math.round(d.confidence*100)+'%')).join(' · ');
 $('osd').textContent=v.osd.length?v.osd.map(r=>r.name+': '+r.text).join(' · '):v.ocr_status;$('accept-recommendation').disabled=!demo&&s.control_blockers.length>0;
 const rec=s.recording;$('record-status').textContent=rec.status;$('record-start').disabled=rec.active||!frameFresh;$('record-stop').disabled=!rec.active;$('record-download').hidden=rec.active||!rec.filename;if(rec.filename)$('record-download').href='/recordings/'+encodeURIComponent(rec.filename);
 map.home=s.home;map.fence=s.limits.radius_m;map.path=s.path;if(t.latitude!==null){map.aircraft={latitude:t.latitude,longitude:t.longitude,heading_deg:t.heading_deg,stale:!posFresh};if(!map.initialised){map.centre={...map.aircraft};map.initialised=true}}map.schedule();
 $('map-coordinate').textContent=posFresh?fmt(t.latitude,6)+' / '+fmt(t.longitude,6):'GPS stale / missing';$('footer-coordinates').textContent='LAT '+fmt(t.latitude,6)+' · LON '+fmt(t.longitude,6);$('footer-limits').textContent=s.limits.radius_m+' m fence · '+s.limits.altitude_m+' m ceiling · '+s.limits.speed_mps+' m/s';
 drawHorizon(t,demo);const canvas=$('overlay'),ctx=canvas.getContext('2d');canvas.width=960;canvas.height=540;ctx.strokeStyle='#c1e66c';ctx.fillStyle='#c1e66c';ctx.lineWidth=3;ctx.font='18px monospace';(v.annotations_embedded ? [] : v.detections).forEach(d=>{const [x,y,x2,y2]=d.box;ctx.strokeRect(x*960,y*540,(x2-x)*960,(y2-y)*540);ctx.fillText(d.label+(v.simulated?' / SIM':'')+' f'+v.detection_sequence,x*960,Math.max(20,y*540-8))});
 $('perf-fps').textContent=fmt(v.fps)+' fps';$('perf-encode').textContent=fmt(v.processing_ms,0)+' ms';$('perf-inference').textContent=v.simulated?'SIM':fmt(v.inference_ms,0)+' ms';$('perf-age').textContent=fmt(v.age_s===null?null:v.age_s*1000,0)+' ms';$('simulation-controls').querySelectorAll('input,select,button').forEach(el=>el.disabled=!demo||(v.backend==='aura'&&el.id==='sim-person'));
 window.dispatchEvent(new CustomEvent('station-state', {detail:s}));
 document.querySelectorAll('#waypoint-rows tr').forEach((row,i)=>row.classList.toggle('current',m.status==='running'&&m.index===i));
 if(Date.now()-lastEventRefresh>3000){lastEventRefresh=Date.now();refreshEvents()}
}
async function refreshEvents(){if(refreshing)return;refreshing=true;try{const events=await request('/api/events');$('activity').replaceChildren();events.slice(0,20).forEach(e=>{const div=document.createElement('div');div.className='event';const strong=document.createElement('strong');strong.textContent=e.kind.replaceAll('_',' ').toUpperCase()+' · '+new Date(e.time).toLocaleTimeString();const p=document.createElement('p');p.textContent=e.payload.note||e.payload.reason||e.payload.name||e.payload.direction||e.payload.policy||e.payload.status||e.payload.mode||'';div.append(strong,p);if(e.payload.snapshot){const a=document.createElement('a');a.href='/snapshots/'+encodeURIComponent(e.payload.snapshot);a.target='_blank';a.rel='noopener';a.textContent='View captured frame ↗';div.append(a)}$('activity').append(div)})}catch(e){toast(e.message,true)}finally{refreshing=false}}
$('refresh').addEventListener('click',safe(async()=>{render(await request('/api/status'));await refreshEvents()}));
setInterval(()=>$('clock').textContent=new Date().toLocaleTimeString(),1000);
function disconnected(){
 online=false;$('connection').textContent='Reconnecting';$('connection-dot').style.background='var(--warning)';$('lease').checked=false;
 $('session-message').textContent='Station disconnected · mission lease will expire';$('video').style.visibility='hidden';$('video').removeAttribute('src');
 document.querySelectorAll('button,input,select,textarea').forEach(el=>{if(!el.disabled){el.dataset.connectionDisabled='true';el.disabled=true}});
 window.dispatchEvent(new Event('station-disconnected'));
}
async function boot(){
 try{
  token=(await request('/api/session')).token;render(await request('/api/status'));updateRows();
  events?.close();events=new EventSource('/api/stream');
  events.onopen=async()=>{try{token=(await request('/api/session')).token;if(!$('video').hasAttribute('src'))$('video').src='/video.mjpg?reconnect='+Date.now();window.dispatchEvent(new Event('station-connected'))}catch(e){disconnected()}};
  events.onmessage=e=>{try{render(JSON.parse(e.data))}catch(error){disconnected();toast('Station data could not be displayed: '+error.message,true)}};
  events.onerror=disconnected;
  await refreshEvents();
 }catch(e){disconnected();setTimeout(boot,1500)}
}
boot();
