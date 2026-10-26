"""Session-local object tracks and guarded geographic estimates."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import time
import uuid

from .control import distance_m
from .geolocation import Calibration, locate
from .intelligence import iou


VEHICLES={'car','van','truck','bus','motor','motorcycle','bicycle','tricycle','awning-tricycle'}


def group(label):
    if label in ('person','pedestrian','people'): return 'person'
    if label in VEHICLES: return 'vehicle'
    return None


def deduplicate(detections):
    valid=[]
    for d in detections:
        box=d.get('box',[])
        if not group(d.get('label')) or len(box)!=4: continue
        try:
            if not all(type(x) in (int,float) and math.isfinite(x) for x in box): continue
            if not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1): continue
            confidence=d.get('confidence')
            if confidence is not None and (type(confidence) not in (int,float) or not 0<=confidence<=1): continue
        except (TypeError,ValueError): continue
        valid.append({**d,'category':group(d['label'])})
    result=[]
    for d in sorted(valid,key=lambda x:x.get('confidence') or 0,reverse=True):
        if not any(d['category']==p['category'] and iou(d['box'],p['box'])>.7 for p in result):
            result.append(d)
    return result


class Observations:
    def __init__(self,store):
        self.store=store
        self.calibration_path=store.root/'camera-calibration.json'
        self.calibration=Calibration()
        self.calibration_error=''
        if self.calibration_path.is_file():
            try: self.calibration=Calibration.parse(json.loads(self.calibration_path.read_text(encoding='utf-8-sig')))
            except (ValueError,TypeError,OSError) as exc: self.calibration_error=str(exc)
        self.session=uuid.uuid4().hex[:8]
        self.next_id=1
        self.tracks={}
        self.source=None
        self.last_sequence=None
        self.reason='Waiting for video'

    def set_calibration(self,data):
        candidate=Calibration.parse(data)
        temporary=self.calibration_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(asdict(candidate),indent=2),encoding='utf-8')
        temporary.replace(self.calibration_path)
        self.calibration=candidate
        self.calibration_error=''
        self.tracks.clear()
        self.last_sequence=None
        self.store.add('mapping_calibration',{'verified':candidate.verified,'source':candidate.source})
        return self.calibration_status()

    def calibration_status(self):
        return {'calibration':asdict(self.calibration),'error':self.calibration_error,
                'status':'Configured estimate; field validation required' if self.calibration.verified else 'Not calibrated',
                'method':'Fixed camera, pinhole lens model, surveyed flat ground; estimated coordinates only'}

    def update(self,video,telemetry,now=None):
        now=time.monotonic() if now is None else now
        source=(video.get('source_id',video.get('source')),telemetry.pose_epoch)
        if source!=self.source:
            self.source=source
            self.tracks.clear()
            self.last_sequence=None
        # Drop fleeting candidates quickly; retain confirmed last-seen observations.
        self.tracks={key:t for key,t in self.tracks.items()
                     if now-t['last_seen'] <= (180 if t['confirmed'] else 3)}
        age=video.get('age_s')
        if age is None or age>1:
            self.reason='Video paused, off or stale; markers are last-seen observations'
            for t in self.tracks.values(): t['visible']=False;t['streak']=0
            return
        sequence=video.get('detection_sequence')
        detections=deduplicate(video.get('detections',[]))
        if sequence is None or sequence==self.last_sequence:
            if not detections:
                for t in self.tracks.values(): t['visible']=False;t['streak']=0
            return
        self.last_sequence=sequence
        pose=None
        self.reason='Location unavailable'
        try:
            if not video.get('live_ready'):
                raise ValueError('Recorded or simulated video cannot be paired with current aircraft GPS')
            if not self.calibration.verified:
                raise ValueError('Camera calibration and surveyed ground are required')
            source_name=str(video.get('source',''))
            if source_name.isdigit(): source_name='camera:'+source_name
            if source_name!=self.calibration.source:
                raise ValueError('This camera has no matching calibration')
            pose=telemetry.pose_at(video.get('captured_monotonic'),self.calibration)
            self.reason='Estimated target locations · surveyed flat-ground model'
        except ValueError as exc:
            self.reason=str(exc)
        used=set()
        was_visible={key:t['visible'] for key,t in self.tracks.items()}
        for t in self.tracks.values(): t['visible']=False
        for d in detections:
            location=None
            reason=self.reason
            if pose is not None:
                try: location=locate(d['box'],pose,self.calibration,video.get('image_size',[0,0]))
                except ValueError as exc: reason=str(exc)
            candidates=[]
            for key,t in self.tracks.items():
                gap=now-t['last_seen']
                if key in used or t['category']!=d['category'] or gap>5: continue
                overlap=iou(t['box'],d['box']) if gap<1.5 else 0
                nearby=location and t['location'] and distance_m(location['latitude'],location['longitude'],
                    t['location']['latitude'],t['location']['longitude'])<3
                if overlap>.2 or nearby:
                    candidates.append((overlap+(0.3 if nearby else 0),key))
            candidates.sort(reverse=True)
            # Do not silently merge two nearby people when association is ambiguous.
            match=candidates[0][1] if candidates and (len(candidates)==1 or candidates[0][0]-candidates[1][0]>.1) else None
            if match is None:
                match=f'{self.session}-{self.next_id:03d}'
                title=f"{d['category'].title()} {self.next_id:03d}"
                self.next_id+=1
                self.tracks[match]=dict(id=match,title=title,category=d['category'],first_seen=now,
                    last_seen=now,streak=0,hits=0,confirmed=False,last_logged=0,location=None,review='unreviewed')
            t=self.tracks[match]
            consecutive=was_visible.get(match,False) and now-t['last_seen']<1
            t.update(box=list(d['box']),label=d.get('original_label',d['label']),confidence=d.get('confidence'),
                     visible=True,last_seen=now,last_seen_utc=datetime.now(timezone.utc).isoformat(),
                     streak=t['streak']+1 if consecutive else 1,hits=t['hits']+1,
                     location=location,location_reason=reason if location is None else 'Estimated ground contact',
                     frame=video.get('source_frame',sequence),source=video.get('source'),
                     source_id=video.get('source_id',video.get('source','native')),
                     association='ambiguous; new track' if candidates and match!=candidates[0][1] else 'local track')
            first_confirmation=not t['confirmed'] and t['streak']>=3
            t['confirmed']=t['confirmed'] or t['streak']>=3
            if t['confirmed'] and (first_confirmation or now-t['last_logged']>=2):
                self.store.add('object_confirmed' if first_confirmation else 'object_update',self.public(t,now))
                t['last_logged']=now
            used.add(match)
        for key,t in self.tracks.items():
            if key not in used: t['streak']=0
        if len(self.tracks)>200:
            self.tracks=dict(sorted(self.tracks.items(),key=lambda kv:kv[1]['last_seen'],reverse=True)[:200])

    def public(self,t,now):
        data={k:v for k,v in t.items() if k not in ('first_seen','last_seen','last_logged','streak')}
        data['age_s']=round(max(0,now-t['last_seen']),1)
        data['status']='tentative' if not t['confirmed'] else 'visible' if t['visible'] and data['age_s']<1 else 'last seen'
        # A tentative candidate never gets a map marker.
        if not t['confirmed']: data['location']=None
        return data

    def snapshot(self,now=None):
        now=time.monotonic() if now is None else now
        items=[self.public(t,now) for t in sorted(self.tracks.values(),key=lambda t:t['last_seen'],reverse=True)]
        confirmed=[t for t in items if t['confirmed'] and t['review']!='dismissed']
        return dict(items=items,confirmed_count=len(confirmed),
                    mapped_count=sum(t['location'] is not None for t in confirmed),
                    reason=self.reason,calibrated=self.calibration.verified,
                    identity_note='IDs are temporary object tracks, not personal identity. Occlusion/crossing may split or swap tracks.')

    def review(self,key,status):
        if status not in ('unreviewed','verified','dismissed') or key not in self.tracks:
            raise ValueError('Observation no longer available or invalid review status')
        self.tracks[key]['review']=status
        item=self.public(self.tracks[key],time.monotonic())
        self.store.add('object_review',item)
        return item
