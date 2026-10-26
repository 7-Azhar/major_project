"""Exercise the running demo integration with real inference; restore settings."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, build_opener, ProxyHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    opener = build_opener(ProxyHandler({}))
    base = f'http://127.0.0.1:{args.port}'
    token = ''

    def get(path):
        with opener.open(base+path, timeout=10) as response:
            return json.load(response)

    def post(path, value=None):
        request = Request(base+path, json.dumps(value or {}).encode(),
                          headers={'Content-Type':'application/json','X-Station-Token':token})
        with opener.open(request, timeout=15) as response:
            return json.load(response)

    def until(predicate, timeout=30):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            value = get('/api/status')
            if predicate(value): return value
            time.sleep(.1)
        raise AssertionError('Timed out waiting for integrated state')

    state = get('/api/status')
    if state['mode'] != 'demo' or state['video'].get('backend') != 'aura':
        raise RuntimeError('Smoke checks require simulated telemetry with the AURA backend')
    if state['mission']['status'] == 'running' or state['recording']['active']:
        raise RuntimeError('Stop the mission and recording before running smoke checks')
    token = get('/api/session')['token']
    saved = get('/vision/api/frame')['config']
    report = {}
    try:
        sources = get('/vision/api/options')['sources']
        source = next(s for s in sources if s.endswith('real_fpv_user_flight.mp4'))
        post('/vision/api/control', dict(source=source, powered=True, playing=True, seek=0,
                                          enhancement='identity', detection='original'))
        first = until(lambda s:s['video'].get('age_s') is not None and s['video']['age_s']<1)
        assert not first['video']['live_ready'], 'Recording must not be live-ready'
        report['real_inference'] = first['video']['aura']['hardware']
        for enhancement, detection in [('bilateral','exclude25'),('zerodce','exclude50')]:
            post('/vision/api/control',dict(enhancement=enhancement,detection=detection))
            result = until(lambda s:s['video']['aura']['config'].get('enhancement')==enhancement and
                           s['video']['aura']['config'].get('detection')==detection and
                           s['video']['age_s'] is not None and s['video']['age_s']<1)
            report[enhancement] = result['video']['inference_ms']
        post('/vision/api/control',dict(enhancement='bilateral',detection='exclude25'))
        until(lambda s:s['video']['aura']['config'].get('enhancement')=='bilateral' and s['video']['age_s'] is not None)
        clip = post('/api/recording/start')
        time.sleep(.7)
        post('/api/recording/stop')
        import cv2
        capture = cv2.VideoCapture(str(Path(__file__).parent/'data'/clip['filename']))
        try:
            ok, frame = capture.read()
            assert ok and frame.size
            report['recording'] = clip['filename']
        finally:
            capture.release()
        incident = post('/api/incidents',dict(note='Integration software smoke test: recorded video, simulated aircraft.'))
        assert incident.get('snapshot')
        report['snapshot'] = incident['snapshot']
        post('/vision/api/control',dict(powered=False))
        off = until(lambda s:s['video']['aura']['config'].get('powered') is False)
        assert off['video']['age_s'] is None and off['video']['detections']==[]
        assert not off['video']['live_ready']
        report['power_off_invalidates_evidence'] = True
        post('/vision/api/control',dict(powered=True,playing=False))
        paused = until(lambda s:s['video']['aura']['config'].get('powered') and not s['video']['aura']['config'].get('playing'))
        assert paused['video']['age_s'] is None and not paused['video']['detections']
        report['pause_invalidates_evidence'] = True
        print(json.dumps(report,indent=2))
    finally:
        post('/vision/api/control',saved)


if __name__ == '__main__':
    main()
