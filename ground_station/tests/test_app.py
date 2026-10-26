import time
import pytest
import cv2
from fastapi.testclient import TestClient
from groundstation.config import Config
from groundstation.main import create_app


def headers(client):
    return {'x-station-token':client.get('/api/session').json()['token']}


def wait_for_frame(client):
    deadline=time.monotonic()+4
    while time.monotonic()<deadline:
        state=client.get('/api/status').json()
        if state['video']['sequence']>0 and state['telemetry']['latitude'] is not None:
            return state
        time.sleep(.05)
    raise AssertionError('Simulator did not produce a frame and telemetry')


def test_demo_end_to_end(tmp_path):
    with TestClient(create_app(Config(),tmp_path)) as client:
        state=wait_for_frame(client)
        assert client.get('/').status_code==200
        assert state['mode']=='demo'
        auth=headers(client)
        before=state['telemetry']['latitude']
        assert client.post('/api/command',json={'text':'forward'},headers=auth).status_code==200
        time.sleep(.7)
        after=client.get('/api/status').json()
        assert after['telemetry']['latitude']>before
        assert after['telemetry']['speed_mps']==0
        incident=client.post('/api/incidents',json={'note':'Possible lead'},headers=auth).json()
        assert incident['drone_position'] is not None
        assert client.get('/snapshots/'+incident['snapshot']).content[:2]==b'\xff\xd8'
        assert client.post('/api/search-grid',json={'latitude':17,'longitude':78},headers=auth).json()['type']=='FeatureCollection'
        assert client.post('/api/release',headers=auth).json()['released']
        assert client.get('/api/export').status_code==200
        assert client.get('/docs').status_code==200
        # Frequent telemetry must not bury the operator's notebook in the UI.
        for i in range(105):
            client.app.state.station.store.add('telemetry',{'sample':i})
        assert any(event['id']==incident['id'] for event in client.get('/api/events').json())


def test_recording_is_decodable(tmp_path):
    with TestClient(create_app(Config(),tmp_path)) as client:
        wait_for_frame(client)
        auth=headers(client)
        started=client.post('/api/recording/start',headers=auth)
        assert started.status_code==200
        filename=started.json()['filename']
        assert client.post('/api/recording/start',headers=auth).status_code==409
        assert client.get('/recordings/'+filename).status_code==409
        time.sleep(.4)
        assert client.post('/api/recording/stop',headers=auth).status_code==200
        assert client.get('/recordings/'+filename).content[:4]==b'RIFF'
        cap=cv2.VideoCapture(str(tmp_path/filename))
        try:
            assert cap.isOpened()
            ok,frame=cap.read()
            assert ok and frame.shape[:2]==(540,960)
        finally:
            cap.release()


def test_write_auth_and_cross_origin(tmp_path):
    with TestClient(create_app(Config(),tmp_path)) as client:
        assert client.post('/api/command',json={'text':'forward'}).status_code==403
        auth={**headers(client),'origin':'https://evil.example'}
        assert client.post('/api/command',json={'text':'forward'},headers=auth).status_code==403
        assert client.get('/api/status',headers={'host':'evil.example'}).status_code==400
        assert client.post('/api/command',json={'text':'arm'},headers=headers(client)).status_code==409


def test_live_control_defaults_disabled(tmp_path):
    with TestClient(create_app(Config(mode='live',video_source='missing.mp4',mavlink_endpoint='udpin:127.0.0.1:14999'),tmp_path)) as client:
        auth=headers(client)
        assert client.post('/api/command',json={'text':'forward'},headers=auth).status_code==409
        assert client.post('/api/lease',json={'enabled':True},headers=auth).status_code==409


def test_invalid_config():
    with pytest.raises(ValueError): Config(mode='live')
