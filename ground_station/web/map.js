export const distance = (a,b) => {
  const rad=Math.PI/180,p1=a.latitude*rad,p2=b.latitude*rad;
  const x=Math.sin((p2-p1)/2)**2+Math.cos(p1)*Math.cos(p2)*Math.sin((b.longitude-a.longitude)*rad/2)**2;
  return 6371000*2*Math.asin(Math.min(1,Math.sqrt(x)));
};
export function project(p,z){const size=256*2**z,lat=Math.max(-85,Math.min(85,p.latitude))*Math.PI/180;return [(p.longitude+180)/360*size,(1-Math.log(Math.tan(lat)+1/Math.cos(lat))/Math.PI)/2*size]}
export function unproject(x,y,z){const size=256*2**z;return {longitude:x/size*360-180,latitude:Math.atan(Math.sinh(Math.PI*(1-2*y/size)))*180/Math.PI}}

export class MissionMap {
  constructor(canvas,{onAdd,onMove,onSelect,onHistory,onCursor}){
    this.canvas=canvas;this.ctx=canvas.getContext('2d');this.callbacks={onAdd,onMove,onSelect,onHistory,onCursor};
    this.centre={latitude:17.385,longitude:78.4867};this.zoom=19;this.layer='offline';this.points=[];this.path=[];this.home=null;this.aircraft=null;this.selected=-1;this.addMode=false;this.editable=true;this.tiles=new Map();this.failures=0;this.queued=false;
    this.observations=[];this.selectedObservation=null;
    new ResizeObserver(()=>this.schedule()).observe(canvas);
    canvas.addEventListener('pointerdown',e=>this.down(e));canvas.addEventListener('pointermove',e=>this.move(e));canvas.addEventListener('pointerup',e=>this.up(e));canvas.addEventListener('pointercancel',()=>{this.drag=null});
    canvas.addEventListener('wheel',e=>{e.preventDefault();this.changeZoom(e.deltaY<0?1:-1)}, {passive:false});
  }
  size(){return [this.canvas.clientWidth,this.canvas.clientHeight]}
  xy(p){const [w,h]=this.size(),c=project(this.centre,this.zoom),v=project(p,this.zoom);return [v[0]-c[0]+w/2,v[1]-c[1]+h/2]}
  point(x,y){const [w,h]=this.size(),c=project(this.centre,this.zoom);return unproject(c[0]+x-w/2,c[1]+y-h/2,this.zoom)}
  eventXY(e){const r=this.canvas.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top]}
  changeZoom(delta){this.zoom=Math.max(3,Math.min(20,this.zoom+delta));this.schedule()}
  down(e){if(e.button!==0)return;const [x,y]=this.eventXY(e);const index=this.points.findIndex(p=>{const q=this.xy(p);return Math.hypot(q[0]-x,q[1]-y)<15});this.drag={x,y,startX:x,startY:y,centre:{...this.centre},index,moved:false};this.canvas.setPointerCapture(e.pointerId);if(index>=0){this.callbacks.onSelect(index);if(this.editable)this.callbacks.onHistory()}}
  move(e){const [x,y]=this.eventXY(e);this.callbacks.onCursor(this.point(x,y));if(!this.drag)return;const d=this.drag;d.moved=d.moved||Math.hypot(x-d.startX,y-d.startY)>3;if(d.index>=0&&this.editable){const p=this.point(x,y);this.callbacks.onMove(d.index,p)}else{const c=project(d.centre,this.zoom);this.centre=unproject(c[0]-(x-d.startX),c[1]-(y-d.startY),this.zoom);this.schedule()}}
  up(e){if(!this.drag)return;const d=this.drag;this.drag=null;if(!d.moved&&d.index<0&&this.addMode&&this.editable){const [x,y]=this.eventXY(e);this.callbacks.onAdd(this.point(x,y))}if(this.canvas.hasPointerCapture(e.pointerId))this.canvas.releasePointerCapture(e.pointerId)}
  fit(){const points=[...this.points,...this.observations.map(t=>t.location),...(this.home?[this.home]:[]),...(this.aircraft?[this.aircraft]:[])];if(!points.length)return;const xs=points.map(p=>p.longitude),ys=points.map(p=>p.latitude);this.centre={latitude:(Math.min(...ys)+Math.max(...ys))/2,longitude:(Math.min(...xs)+Math.max(...xs))/2};const [w,h]=this.size();for(let z=20;z>=3;z--){const values=points.map(p=>project(p,z));const width=Math.max(...values.map(p=>p[0]))-Math.min(...values.map(p=>p[0])),height=Math.max(...values.map(p=>p[1]))-Math.min(...values.map(p=>p[1]));if(width<w*.7&&height<h*.65){this.zoom=z;break}}this.schedule()}
  schedule(){if(this.queued)return;this.queued=true;requestAnimationFrame(()=>{this.queued=false;this.draw()})}
  tile(z,x,y){const count=2**z;if(y<0||y>=count)return null;x=((x%count)+count)%count;const key=`${this.layer}/${z}/${x}/${y}`;if(this.tiles.has(key))return this.tiles.get(key);const img=new Image();const tile={img,ready:false};img.onload=()=>{tile.ready=true;this.schedule()};img.onerror=()=>{this.failures++;this.schedule()};img.src=this.layer==='satellite'?`https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`:`https://tile.openstreetmap.org/${z}/${x}/${y}.png`;this.tiles.set(key,tile);if(this.tiles.size>128)this.tiles.delete(this.tiles.keys().next().value);return tile}
  draw(){const [w,h]=this.size();if(!w||!h)return;const dpr=Math.min(2,window.devicePixelRatio||1);if(this.canvas.width!==w*dpr||this.canvas.height!==h*dpr){this.canvas.width=w*dpr;this.canvas.height=h*dpr}const ctx=this.ctx;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#1b292d';ctx.fillRect(0,0,w,h);const c=project(this.centre,this.zoom),left=c[0]-w/2,top=c[1]-h/2;
    ctx.strokeStyle='#30444b';ctx.lineWidth=1;const step=64;for(let x=-left%step;x<w;x+=step){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}for(let y=-top%step;y<h;y+=step){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
    if(this.layer!=='offline'){const z=Math.min(19,this.zoom),scale=2**(this.zoom-z),tileSize=256*scale;for(let x=Math.floor(left/tileSize);x<=Math.floor((left+w)/tileSize);x++)for(let y=Math.floor(top/tileSize);y<=Math.floor((top+h)/tileSize);y++){const tile=this.tile(z,x,y);if(tile?.ready)ctx.drawImage(tile.img,x*tileSize-left,y*tileSize-top,tileSize,tileSize)}}
    const mpp=156543.03392*Math.cos(this.centre.latitude*Math.PI/180)/2**this.zoom;
    if(this.home){const [x,y]=this.xy(this.home);ctx.beginPath();ctx.arc(x,y,(this.fence||50)/mpp,0,Math.PI*2);ctx.setLineDash([6,6]);ctx.strokeStyle='#d5bd6680';ctx.lineWidth=1;ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#d2c685';ctx.fillRect(x-5,y-5,10,10);this.label('HOME',x+9,y-12,'#e8e8d6')}
    this.line(this.path,'#6dbcd4',1.5);this.line(this.points,'#e5df71',2.5);
    this.points.forEach((p,i)=>{const [x,y]=this.xy(p);ctx.beginPath();ctx.arc(x,y,i===this.selected?12:10,0,Math.PI*2);ctx.fillStyle=i===this.selected?'#e8f4ac':'#91cf47';ctx.fill();ctx.strokeStyle='#f0f8d5';ctx.lineWidth=1;ctx.stroke();ctx.fillStyle='#203318';ctx.font='bold 10px monospace';ctx.textAlign='center';ctx.fillText(i+1,x,y+3.5);ctx.textAlign='left'});
    this.observations.forEach(t=>{
      const [x,y]=this.xy(t.location),color=t.status==='last seen'?'#9caeb3':t.category==='person'?'#c3dc8c':'#efb978';
      ctx.save();ctx.beginPath();ctx.arc(x,y,t.location.error_radius_m/mpp,0,2*Math.PI);ctx.fillStyle=color+'18';ctx.fill();ctx.strokeStyle=color+'88';ctx.setLineDash([4,4]);ctx.stroke();ctx.setLineDash([]);
      ctx.beginPath();ctx.arc(x,y,this.selectedObservation===t.id?9:6,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#eef4ef';ctx.stroke();
      this.label(t.title+(t.status==='last seen'?' · last seen':''),x+12,y-12,color);ctx.restore();
    });
    if(this.aircraft){const [x,y]=this.xy(this.aircraft);ctx.save();ctx.translate(x,y);ctx.rotate((this.aircraft.heading_deg||0)*Math.PI/180);ctx.beginPath();ctx.moveTo(0,-15);ctx.lineTo(10,11);ctx.lineTo(0,6);ctx.lineTo(-10,11);ctx.closePath();ctx.fillStyle=this.aircraft.stale?'#859199':'#77c9f1';ctx.fill();ctx.strokeStyle='#e9f8ff';ctx.stroke();ctx.restore()}
    ctx.fillStyle='#d9e4e7';ctx.font='11px monospace';ctx.fillText('N ↑',w-53,h-30);document.getElementById('map-scale').textContent=`${Math.round(75*mpp)} m`;this.attribution();
  }
  line(points,color,width){if(!points.length)return;const ctx=this.ctx;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();points.forEach((p,i)=>{const [x,y]=this.xy(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}
  label(text,x,y,color){const ctx=this.ctx;ctx.font='9px monospace';const width=ctx.measureText(text).width;ctx.fillStyle='#172228dd';ctx.fillRect(x-4,y-11,width+8,16);ctx.fillStyle=color;ctx.fillText(text,x,y)}
  attribution(){const el=document.getElementById('map-attribution');const key=this.layer+this.failures;if(el.dataset.key===key)return;el.dataset.key=key;el.replaceChildren();if(this.layer==='offline'){el.textContent='Offline tactical grid · no terrain data';return}const a=document.createElement('a');a.target='_blank';a.rel='noopener';a.textContent=this.layer==='satellite'?'Esri World Imagery · Esri, Maxar, Earthstar Geographics & GIS community':'© OpenStreetMap contributors';a.href=this.layer==='satellite'?'https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9':'https://www.openstreetmap.org/copyright';el.append(a);if(this.failures)el.append(' · Some tiles unavailable')}
}
