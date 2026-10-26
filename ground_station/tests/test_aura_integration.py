"""Contract tests for shared video, source freshness and same-origin controls."""
import base64
from copy import deepcopy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from groundstation.aura import AuraVideo
from groundstation.config import Config
from groundstation.intelligence import Intelligence
from groundstation.main import create_app
from groundstation.station import Station


def packet():
    _, jpg = cv2.imencode('.jpg', np.zeros((100, 200, 3), np.uint8))
    return dict(server_id='server', sequence=1, source_epoch='first', frame=3,
        status='playing', captured_at=1000., width=200, height=100, live=True,
        output=base64.b64encode(jpg).decode(),
        config=dict(source='camera:0', playing=True, powered=True, detection='original', enhancement='identity'),
        detections=[dict(label='pedestrian', confidence=.8, xyxy=[20,10,60,90]),
                    dict(label='car', confidence=.7, xyxy=[100,20,180,80])])


def test_mapping_and_capture_age_are_preserved():
    video = AuraVideo(Config(video_backend='aura'))
    video.accept(packet(), now=100, wall=1000.2)
    state = video.snapshot(now=100.3)
    assert state['live_ready']
    assert state['age_s'] == pytest.approx(.5)
    person, car = state['detections']
    assert person['label'] == 'person' and person['original_label'] == 'pedestrian'
    assert person['box'] == [.1,.1,.3,.9] and car['label'] == 'car'
    assert video.jpeg[:2] == b'\xff\xd8' and video.frame.shape == (100,200,3)
    assert not video.snapshot(now=102)['detections']
    assert not video.snapshot(now=102)['live_ready']


def test_repeated_controls_do_not_confirm_person_and_new_source_resets():
    video, intelligence = AuraVideo(Config(video_backend='aura')), Intelligence()
    p = packet()
    for sequence in range(1, 5):
        p['sequence'] = sequence
        video.accept(deepcopy(p), now=100, wall=1000)
        intelligence.update(video.snapshot(now=100), now=100)
    assert intelligence.snapshot(now=100)['confirmed_count'] == 0
    assert intelligence.tracks[0]['hits'] == 1
    for frame in (4,5):
        p['frame'] = frame
        video.accept(deepcopy(p), now=100, wall=1000)
        intelligence.update(video.snapshot(now=100), now=100)
    assert intelligence.snapshot(now=100)['confirmed_count'] == 1
    p['source_epoch'] = 'second'
    video.accept(p, now=100, wall=1000)
    intelligence.update(video.snapshot(now=100), now=100)
    assert intelligence.snapshot(now=100)['confirmed_count'] == 0


@pytest.mark.parametrize('case', ['off','paused','error','ended','old_server','future_clock','recording','disconnected'])
def test_nonlive_inputs_never_enable_control(case):
    video = AuraVideo(Config(video_backend='aura'))
    p = packet()
    if case == 'off': p['config']['powered'] = False
    if case == 'paused': p['config']['playing'] = False
    if case in ('error','ended'): p['status'] = case
    if case == 'old_server': p.pop('captured_at')
    if case == 'future_clock': p['captured_at'] = 2000
    if case == 'recording': p['config']['source'], p['live'] = 'sample.mp4', False
    video.accept(p, now=100, wall=1000)
    if case == 'disconnected': video.disconnect('test')
    state = video.snapshot(now=100)
    assert not state['live_ready']
    if case != 'recording': assert not state['detections']


def test_recording_blocks_flight_even_with_fresh_packet(tmp_path):
    station = Station(Config(mode='live', video_backend='aura', live_control=True), tmp_path)
    try:
        p = packet()
        p['captured_at'] = time.time()
        p['config']['source'], p['live'] = 'sample.mp4', False
        station.video.accept(p)
        assert any('recordings cannot' in reason for reason in station.blockers())
    finally:
        station.close()


@pytest.fixture
def upstream():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            payload = packet()
            payload['captured_at'] = time.time()
            if self.path.startswith('/api/options'): payload = {'sources': ['test.mp4'], 'enhancements': {'identity':'Original'}}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError): pass
            time.sleep(.05)
        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            requests.append((self.path,body))
            data = b'{"revision":1,"server_id":"server","source":"test.mp4"}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_port, requests
    server.shutdown()
    server.server_close()
    thread.join()


def test_proxy_auth_controls_upload_and_ui(upstream, tmp_path):
    port, requests = upstream
    with TestClient(create_app(Config(video_backend='aura', aura_port=port), tmp_path)) as client:
        token = client.get('/api/session').json()['token']
        auth = {'x-station-token':token}
        assert token in client.get('/vision/').text
        assert client.get('/vision/app.js').status_code == 200
        assert 'Vision workspace' in client.get('/').text
        assert client.get('/vision/api/options').json()['sources'] == ['test.mp4']
        assert client.post('/vision/api/control',json={'powered':False}).status_code == 403
        assert client.post('/vision/api/control',json={'powered':False},headers={**auth,'origin':'https://bad.example'}).status_code == 403
        assert client.post('/vision/api/control',json={'powered':False},headers=auth).status_code == 200
        assert requests[-1][0] == '/api/control' and json.loads(requests[-1][1]) == {'powered':False}
        assert client.post('/vision/api/upload', content=b'fake-video', headers={**auth,'content-type':'application/octet-stream','x-filename':'test.mp4'}).status_code == 200
        assert requests[-1] == ('/api/upload',b'fake-video')
        assert client.get('/vision/api/unknown').status_code == 404
        assert client.post('/api/simulation', json={'person_visible':True}, headers=auth).status_code == 409


def test_proxy_unavailable_is_actionable(tmp_path):
    with TestClient(create_app(Config(video_backend='aura',aura_port=65534),tmp_path)) as client:
        response = client.get('/vision/api/options')
        assert response.status_code == 503 and 'Start AURA' in response.json()['error']
