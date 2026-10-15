// Keep input selection separate from the processing controls.
const inputCard=document.createElement('section');
inputCard.className='card';
inputCard.innerHTML=`<div class="eyebrow">Video input</div><h2>Choose your source</h2>
<label for="videoFile">Video from this PC</label><input id="videoFile" type="file" accept=".mp4,.mov,.avi,.mkv,.webm,.m4v" style="width:100%;font-size:12px">
<p class="muted">Up to 1 GB. A copy stays in AURA’s local uploads folder; nothing is sent to the cloud.</p><div id="uploadStatus" class="status" role="status"></div>
<label for="cameraIndex">Webcam / USB capture device</label><select id="cameraIndex">${Array.from({length:10},(_,i)=>`<option value="${i}">Camera ${i}${i===0?' (usually built-in)':''}</option>`).join('')}</select>
<div style="display:flex;gap:8px;margin-top:10px"><button id="connectCamera">Connect</button><button id="disconnectCamera">Disconnect</button></div>
<p class="muted">Select the device number and connect. For USB devices try Camera 1 or 2. Numbers depend on Windows; this list is not a device scan. Close other camera apps if connection fails.</p>`;
document.querySelector('aside').prepend(inputCard);
$('connectCamera').onclick=()=>control({source:'camera:'+$('cameraIndex').value,playing:true});
$('disconnectCamera').onclick=()=>control({source:options.sources[0],playing:false});
$('videoFile').onchange=()=>{
 const file=$('videoFile').files[0];if(!file)return;
 if(file.size>1024**3){$('uploadStatus').textContent='Video is larger than 1 GB.';return}
 $('videoFile').disabled=true;
 const request=new XMLHttpRequest();request.open('POST','/api/upload');request.setRequestHeader('Content-Type','application/octet-stream');request.setRequestHeader('X-Filename',encodeURIComponent(file.name));
 request.upload.onprogress=e=>{$('uploadStatus').textContent=e.lengthComputable?'Copying locally: '+Math.round(e.loaded/e.total*100)+'%':'Copying locally…'};
 request.onerror=()=>{$('uploadStatus').textContent='Upload failed. Check the local server.';$('videoFile').disabled=false};
 request.onload=async()=>{try{const result=JSON.parse(request.responseText);if(request.status!==200)throw Error(result.error);options.sources.push(result.source);$('source').add(new Option(file.name,result.source));$('source').value=result.source;await control({source:result.source,playing:false});$('uploadStatus').textContent='Ready: '+file.name;}catch(e){$('uploadStatus').textContent=e.message}finally{$('videoFile').disabled=false;$('videoFile').value=''}};
 request.send(file);
};
// Reflect the active backend source, including live input, without inventing device names.
setInterval(()=>{if(!state)return;const live=state.config.source.startsWith('camera:');
 if(live&&!Array.from($('source').options).some(o=>o.value===state.config.source))$('source').add(new Option('Live '+state.config.source,state.config.source));
 $('source').value=state.config.source;
 $('disconnectCamera').disabled=!live;
 if(live){$('seek').disabled=true;$('restart').disabled=true;$('time').textContent='LIVE';$('play').textContent=state.config.playing?'Pause camera':'Resume camera';}
},100);
