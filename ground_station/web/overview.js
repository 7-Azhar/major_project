import {MissionMap} from './map.js';
const $=id=>document.getElementById(id);
let token='',state=null,online=false,system=null,events=null,server=null,calibration=null,toastTimer,rowsKey='';
let sample=null, leasePending=false;
const fmt=(n,d=1)=>Number.isFinite(n)?n.toFixed(d):'—';
function toast(message,error=false){$('toast').textContent=message;$('toast').classList.toggle('error',error);$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,6500)}
async function request(path,body){
 const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Station-Token':token},body:JSON.stringify(body)};
 let response=await fetch(path,{...options,signal:AbortSignal.timeout(10000)});
 if(response.status===403&&body!==undefined){token=(await request('/api/session')).token;options.headers['X-Station-Token']=token;response=await fetch(path,{...options,signal:AbortSignal.timeout(10000)})}
 const value=await response.json();if(!response.ok)throw Error(typeof value.detail==='string'?value.detail:JSON.stringify(value.detail||value.error));return value;
}
const safe=fn=>async(...args)=>{try{await fn(...args)}catch(e){toast(e.message,true)}};
const map=new MissionMap($('map'),{onAdd:()=>{},onMove:()=>{},onSelect:()=>{},onHistory:()=>{},onCursor:()=>{}});
map.editable=false;
$('zoom-in').onclick=()=>map.changeZoom(1);$('zoom-out').onclick=()=>map.changeZoom(-1);
$('map-centre').onclick=()=>{if(map.aircraft){map.centre={...map.aircraft};map.schedule()}else toast('Real aircraft GPS is not available yet')};
$('fit-detections').onclick=()=>{if(map.observations.length||map.aircraft)map.fit();else toast('No estimated object locations or real aircraft GPS yet')};
$('map-layer').onchange=()=>{map.layer=$('map-layer').value;map.failures=0;map.schedule();if(map.layer!=='offline')toast('Online tiles show the selected map area; object coordinates are estimated separately')};
function filtered(){return (sample?.items||state?.observations?.items||[]).filter(t=>($('category').value==='all'||t.category===$('category').value)&&($('show-dismissed').checked||t.review!=='dismissed'))}
$('sample-map').onclick=safe(async()=>{
 if(sample){sample=null;map.aircraft=null;map.initialised=false}
 else{$('lease').checked=false;await request('/api/release',{});sample=await request('/api/observations/example')}
 $('sample-map').textContent=sample?'Exit sample preview':'Preview sample locations';$('sample-notice').hidden=!sample;
 rowsKey='';if(state)render(state);if(sample)map.fit();
});
function renderObjects(){
 const items=filtered();map.observations=items.filter(t=>t.confirmed&&t.location&&t.review!=='dismissed');map.schedule();
 $('map-empty').hidden=map.observations.length>0;
 const key=JSON.stringify(items.map(t=>[t.id,t.status,t.label,t.review,t.confidence,t.location,t.location_reason,Math.floor(t.age_s)]));
 if(key===rowsKey)return;rowsKey=key;$('objects').replaceChildren();
 if(!items.length){const p=document.createElement('p');p.className='empty-observations';p.textContent='No observations in this view. Play video in Vision controls. Unlocated objects will appear here even before GPS is connected.';$('objects').append(p);return}
 items.slice(0,60).forEach(t=>{
  const row=document.createElement('article');row.className='object-row';const text=document.createElement('div');
  const title=document.createElement('h3');title.textContent=t.title+' · '+t.label;const status=document.createElement('p');
  status.textContent=(sample?'Synthetic illustration':t.status==='tentative'?'Checking across frames':t.status==='last seen'?'Last seen '+Math.floor(t.age_s)+'s ago':'Visible now')+' · '+(t.confidence===null?'confidence unavailable':Math.round(t.confidence*100)+'%')+' · '+t.review;
  const location=document.createElement('p');location.className=t.location?'location':'dim';location.textContent=t.location?
    'Estimated '+fmt(t.location.latitude,6)+', '+fmt(t.location.longitude,6)+' · error radius ~'+fmt(t.location.error_radius_m)+' m':
    'Location unavailable · '+(t.status==='tentative'?'waiting for repeated detection':t.location_reason);
  text.append(title,status,location);const buttons=document.createElement('div');buttons.className='object-actions';
  function button(label,fn,disabled=false){const b=document.createElement('button');b.textContent=label;b.disabled=disabled||!online;b.onclick=safe(fn);buttons.append(b)}
  button('Locate',()=>{map.centre={...t.location};map.selectedObservation=t.id;map.schedule();$('map').scrollIntoView({behavior:'smooth',block:'center'})},!t.location||t.review==='dismissed');
  button(t.review==='verified'?'Verified ✓':'Verify',async()=>{await request('/api/observations/'+t.id+'/review',{status:'verified'});toast('Observation marked as reviewed')},!t.confirmed||!!sample);
  button(t.review==='dismissed'?'Restore':'Dismiss',async()=>{await request('/api/observations/'+t.id+'/review',{status:t.review==='dismissed'?'unreviewed':'dismissed'})},!!sample);
  row.append(text,buttons);$('objects').append(row);
 });
}
$('category').onchange=()=>{rowsKey='';renderObjects()};$('show-dismissed').onchange=$('category').onchange;
function render(s){
 if(server!==s.server_id){server=s.server_id;$('lease').checked=false;system=null;void loadSystem();rowsKey=''}
 state=s;online=true;$('connection').textContent='Station online';
 const t=s.telemetry,v=s.video,live=s.mode==='live',gpsFresh=live&&t.fix_type>=3&&(t.ages_s.position??Infinity)<2.5&&(t.ages_s.gps??Infinity)<2.5;
 $('context').textContent=s.mission.status==='running'?'Advanced mission is running. Release laptop control to pause it.':live?'Live telemetry · object locations are estimates, with uncertainty':'Hardware pending · real video review, no real aircraft GPS';
 $('aircraft-position').textContent=gpsFresh?fmt(t.latitude,6)+', '+fmt(t.longitude,6):'Aircraft position unavailable';
 $('map-label').textContent=gpsFresh?'AIRCRAFT GPS · NOT OBJECT GPS':'WAITING FOR REAL GPS';
 if(gpsFresh){map.aircraft={latitude:t.latitude,longitude:t.longitude,heading_deg:t.heading_deg,stale:false};if(!map.initialised){map.centre={...map.aircraft};map.initialised=true}}
 else if(map.aircraft){map.aircraft.stale=true}
 $('mapping-state').textContent=s.observations.reason;$('count-summary').textContent=s.observations.confirmed_count+' tracked · '+s.observations.mapped_count+' located';
 if(sample){map.aircraft={...sample.aircraft,heading_deg:sample.aircraft.yaw_deg};$('aircraft-position').textContent='Synthetic aircraft · not a real GPS fix';$('map-label').textContent='SAMPLE MAP';$('mapping-state').textContent=sample.reason;$('count-summary').textContent='2 sample locations';}
 document.querySelectorAll('a[href^="/api/observations/"]').forEach(a=>{a.hidden=!!sample});
 $('telemetry-mode').textContent=live?'MAVLINK':'HARDWARE PENDING';
 $('gps').textContent=gpsFresh?'3D':'—';$('satellites').textContent=gpsFresh?(t.satellites??'—')+' satellites':'No fresh real GPS';
 $('battery').textContent=live&&(t.ages_s.battery??Infinity)<2.5?fmt(t.battery_percent,0)+'%':'—';
 $('flight-mode').textContent=live?t.mode:'Firmware not confirmed';
 $('altitude').textContent=gpsFresh?fmt(t.altitude_m)+' m':'—';$('speed').textContent=gpsFresh?fmt(t.speed_mps):'—';
 $('telemetry-status').textContent=live?t.status:'Simulated telemetry is kept out of object GPS. Connect hardware after confirming its firmware.';
 const fresh=v.age_s!==null&&v.age_s<1;
 $('video').style.visibility=v.backend==='aura'&&(!v.connected||!v.aura.config.powered)?'hidden':'visible';
 $('video-badge').textContent=v.backend==='aura'&&!v.aura.config.powered?'OFF':fresh?'PLAYING':'PAUSED / STALE';
 $('record-start').disabled=s.recording.active||!fresh;$('record-stop').disabled=!s.recording.active;$('record-status').textContent=s.recording.status;
 $('record-download').hidden=!s.recording.filename||s.recording.active;if(s.recording.filename)$('record-download').href='/recordings/'+encodeURIComponent(s.recording.filename);
 const people=s.observations.items.filter(o=>o.category==='person'&&o.confirmed&&o.status==='visible'&&o.review!=='dismissed');
 $('suggestion').textContent=people.length?people.length+' persistent person track(s). Check the scene; a brief hold may help review.':'No persistent person requiring a hold. Operator remains in control.';
 const allowed=!sample&&live&&s.control_blockers.length===0;
 $('lease').disabled=!!sample||!live||!system?.live_control;
 if(!s.lease_active&&live&&!leasePending)$('lease').checked=false;
 document.querySelectorAll('[data-command]').forEach(b=>b.disabled=!allowed);
 $('assist-hold').disabled=!allowed||!people.length;
 $('release').disabled=false;$('note-open').disabled=false;
 $('control-status').textContent=live?(s.control_blockers.join(' · ')||'Ready for a brief operator-approved request'):'Movement unavailable until the flight controller and real telemetry are connected.';
 renderObjects();
 window.dispatchEvent(new CustomEvent('station-state',{detail:s}));
}
async function loadSystem(){try{system=await request('/api/system')}catch(e){system=null}}
async function move(direction){if(!online||state?.mode!=='live')throw Error('Real aircraft connection required');await request('/api/command',{text:direction,duration_s:.5});toast('Requested '+direction+' for 0.5 s; verify aircraft response')}
document.querySelectorAll('[data-command]').forEach(b=>b.onclick=safe(()=>move(b.dataset.command)));
$('assist-hold').onclick=safe(()=>move('hold'));
$('lease').onchange=safe(async()=>{leasePending=true;try{await request('/api/lease',{enabled:$('lease').checked})}catch(e){$('lease').checked=false;throw e}finally{leasePending=false}});
$('release').onclick=safe(async()=>{$('lease').checked=false;await request('/api/release',{});toast('Laptop movement released')});
setInterval(async()=>{if(online&&$('lease').checked&&!document.hidden){try{await request('/api/lease',{enabled:true})}catch(e){$('lease').checked=false;toast(e.message,true)}}},1000);
document.addEventListener('visibilitychange',()=>{if(document.hidden){$('lease').checked=false;void request('/api/lease',{enabled:false}).catch(()=>{})}});
$('record-start').onclick=safe(async()=>{await request('/api/recording/start',{});toast('Recording started')});
$('record-stop').onclick=safe(async()=>{const r=await request('/api/recording/stop',{});toast(r.status)});
$('note-open').onclick=()=>{$('note-dialog').showModal();$('note').focus()};$('note-close').onclick=()=>$('note-dialog').close();
$('note-form').onsubmit=safe(async e=>{e.preventDefault();const r=await request('/api/incidents',{note:$('note').value});$('note-dialog').close();$('note').value='';toast(r.snapshot?'Note and displayed frame saved':'Note saved; no image available')});
async function setup(){
 const result=await request('/api/mapping/calibration');calibration=result.calibration;$('setup-status').replaceChildren();
 for(const line of [result.status,result.error,system?.mode==='live'?'MAVLink endpoint: '+system.mavlink_endpoint:'Current mode: video review / hardware pending',
  'Mapping: fixed analog camera · '+calibration.source,'Live movement: '+(system?.live_control?'configured; telemetry gates still apply':'disabled')].filter(Boolean)){
  const p=document.createElement('p');p.textContent=line;$('setup-status').append(p)
 }
 $('calibration-current').textContent=JSON.stringify(calibration,null,2);
}
$('setup-open').onclick=safe(async()=>{await loadSystem();await setup();$('setup-dialog').showModal()});$('setup-close').onclick=()=>$('setup-dialog').close();
$('calibration-download').onclick=()=>{
 const data={...calibration,verified:false};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='camera-calibration.template.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 toast('Template values are examples. Replace them with measurements before setting verified to true.');
};
$('calibration-file').onchange=safe(async()=>{
 try{const file=$('calibration-file').files[0];if(!file)return;if(file.size>20000)throw Error('Calibration JSON must be under 20 KB');
  await request('/api/mapping/calibration',JSON.parse(await file.text()));await setup();$('calibration-feedback').textContent='Calibration saved. Existing session tracks reset to avoid mixing coordinate systems.';
 }finally{$('calibration-file').value=''}
});
function disconnected(){
 online=false;$('connection').textContent='Reconnecting…';$('lease').checked=false;rowsKey='';
 $('video').style.visibility='hidden';
 $('video').removeAttribute('src');
 for(const id of ['lease','assist-hold','record-start','record-stop','note-open','release'])$(id).disabled=true;
 document.querySelectorAll('[data-command],.object-actions button').forEach(b=>b.disabled=true);
 $('context').textContent='Station disconnected · displayed observations are historical';
 $('control-status').textContent='Connection lost. The movement lease will expire.';
 if(map.aircraft)map.aircraft.stale=true;map.observations=map.observations.map(t=>({...t,status:'last seen'}));map.schedule();
 window.dispatchEvent(new Event('station-disconnected'));
}
async function boot(){
 try{token=(await request('/api/session')).token;await loadSystem();render(await request('/api/status'));
  events?.close();events=new EventSource('/api/stream');
  events.onopen=async()=>{try{token=(await request('/api/session')).token;await loadSystem();if(!$('video').hasAttribute('src'))$('video').src='/video.mjpg?reconnect='+Date.now();window.dispatchEvent(new Event('station-connected'))}catch(e){disconnected()}};
  events.onmessage=e=>{try{render(JSON.parse(e.data))}catch(error){disconnected();toast(error.message,true)}};
  events.onerror=disconnected;
 }catch(e){disconnected();setTimeout(boot,1500)}
}
boot();
