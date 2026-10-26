"""Temporal image-space evidence; no person geolocation is inferred."""
import time


def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1])
    x2,y2=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/max(.000001,union)


class Intelligence:
    def __init__(self):
        self.policy = 'advisory'
        self.tracks = []
        self.next_id = 1
        self.last_sequence = None
        self.source_id = None
        self.last_auto_incident = 0
        self.decision = 'Monitoring; operator remains in control'

    def update(self,video,now=None):
        now = time.monotonic() if now is None else now
        if video.get('source_id') != self.source_id:
            self.tracks = []
            self.last_sequence = None
            self.source_id = video.get('source_id')
        self.tracks = [t for t in self.tracks if now-t['seen_at'] < 1.5]
        sequence = video.get('detection_sequence')
        if video.get('age_s') is None or video['age_s'] > 1:
            self.tracks = []
            self.decision = 'Video stale; mission should pause'
            return
        people = [d for d in video.get('detections',[]) if d.get('label') == 'person' and 'box' in d]
        if sequence is None or not people:
            self.tracks = []
        if sequence is None or sequence == self.last_sequence:
            # Settings can invalidate detections without producing a new frame.
            self.tracks = [t for t in self.tracks if any(iou(t['box'],d['box'])>.25 for d in people)]
            self.update_decision(now)
            return
        self.last_sequence = sequence
        matched = set()
        for d in people:
            best = max((t for t in self.tracks if t['id'] not in matched),key=lambda t:iou(t['box'],d['box']),default=None)
            if best and iou(best['box'],d['box'])>.25:
                best.update(box=d['box'], seen_at=now, hits=best['hits']+1, confidence=d.get('confidence'))
            else:
                best = {'id':self.next_id,'box':d['box'],'seen_at':now,'hits':1,
                        'confidence':d.get('confidence'),'simulated':d.get('detector')=='SIMULATED annotation'}
                self.next_id += 1
                self.tracks.append(best)
            matched.add(best['id'])
        # Confirmation requires consecutive observed detection frames.
        self.tracks = [track for track in self.tracks if track['id'] in matched]
        self.update_decision(now)

    def update_decision(self, now):
        confirmed = [t for t in self.tracks if t['hits']>=3 and now-t['seen_at']<.8]
        self.decision = f'{len(confirmed)} persistent person candidate(s); verify the scene' if confirmed else 'No persistent person candidate; continue observation'

    def snapshot(self,now=None):
        now = time.monotonic() if now is None else now
        return {'policy':self.policy,'decision':self.decision,
                'tracks':[{k:v for k,v in t.items() if k!='seen_at'} for t in self.tracks if now-t['seen_at']<1.5],
                'confirmed_count':sum(t['hits']>=3 and now-t['seen_at']<.8 for t in self.tracks),
                'engine':'Local intent planner + temporal vision policy (no language model)'}
