"""Regressions discovered in the integration audit."""
import json
import time
from unittest.mock import Mock

import pytest

from groundstation.aura import AuraVideo
from groundstation.config import Config, config_fingerprint
from groundstation.control import control_blockers
from groundstation.intelligence import Intelligence
from groundstation.missions import Mission
from groundstation.recording import Recorder
from groundstation.station import Station
from groundstation.storage import Store
from groundstation.video import Video
from test_aura_integration import packet
from test_control import healthy, live_config
from test_missions import point


@pytest.mark.parametrize('values', [
    {'live_control':'false'}, {'live_missions':1}, {'video_source':True},
    {'detection_interval_s':float('nan')}, {'ocr_interval_s':float('inf')},
    {'confidence_threshold':'0.5'}, {'video_width':640.5}, {'jpeg_quality':101},
    {'target_system':0}, {'mavlink_endpoint':''}, {'osd_regions':{}},
    {'osd_regions':[{'name':'test','box':[.9,0,.5,.1]}]},
])
def test_invalid_config_is_rejected(values):
    with pytest.raises(ValueError): Config(**values)


def test_config_load_is_strict_and_windows_bom_supported(tmp_path, monkeypatch):
    path = tmp_path/'config.json'
    monkeypatch.setenv('GS_CONFIG',str(path))
    with pytest.raises(ValueError,match='not found'): Config.load()
    path.write_text('[]')
    with pytest.raises(ValueError,match='object'): Config.load()
    path.write_text('{"mode":"demo"}',encoding='utf-8-sig')
    assert Config.load().mode == 'demo'
    assert config_fingerprint(Config()) != config_fingerprint(Config(mode='live',video_source=0))


@pytest.mark.parametrize('change', [{'latitude':float('nan')}, {'longitude':float('inf')},
    {'battery_percent':float('nan')}, {'timestamps':{}}])
def test_nonfinite_telemetry_cannot_pass_control_checks(change):
    assert control_blockers(live_config(),{**healthy(),**change},True,.1,now=101)


@pytest.mark.parametrize('age',[float('nan'),float('inf'),-1])
def test_invalid_video_age_blocks_control(age):
    assert control_blockers(live_config(),healthy(),True,age,now=101)


def test_waypoint_hold_restarts_after_leaving_target():
    config = Config()
    mission = Mission()
    mission.load('Hold',[point(config,hold=2)])
    mission.start()
    mission.tick(point(config),.5,100)
    mission.tick(point(config,north=5),.5,101)
    mission.tick(point(config),.5,102)
    assert mission.status == 'running'
    mission.tick(point(config),.5,104)
    assert mission.status == 'completed'


def test_running_mission_cannot_restart_and_new_plan_clears_distance():
    mission = Mission()
    mission.load('First',[point(Config(),north=10)])
    mission.start()
    mission.tick(point(Config()),.5,100)
    with pytest.raises(ValueError,match='already running'): mission.start()
    mission.pause('Edit')
    mission.load('Second',[point(Config(),north=20)])
    assert mission.distance_to_target is None


def test_tracker_missing_detection_breaks_confirmation():
    ai=Intelligence()
    p={'age_s':.1,'detection_sequence':1,'detections':[{'label':'person','box':[.1,.1,.3,.9]}]}
    ai.update(p,100)
    ai.update({**p,'detection_sequence':2},100.1)
    ai.update({**p,'detection_sequence':3,'detections':[]},100.2)
    ai.update({**p,'detection_sequence':4},100.3)
    assert ai.snapshot(100.3)['confirmed_count']==0
    assert ai.tracks[0]['hits']==1


def test_tracker_invalidated_detections_clear_on_same_frame():
    ai=Intelligence()
    p={'age_s':.1,'detections':[{'label':'person','box':[.1,.1,.3,.9]}]}
    for sequence in (1,2,3):
        ai.update({**p,'detection_sequence':sequence},100+sequence*.1)
    assert ai.snapshot(100.3)['confirmed_count']==1
    ai.update({**p,'detection_sequence':3,'detections':[]},100.4)
    assert ai.snapshot(100.4)['confirmed_count']==0
    assert not ai.tracks
    assert ai.decision.startswith('No persistent')


def test_tracker_duplicate_frame_does_not_extend_confirmation():
    ai=Intelligence()
    p={'age_s':.1,'detections':[{'label':'person','box':[.1,.1,.3,.9]}]}
    for sequence in (1,2,3):
        ai.update({**p,'detection_sequence':sequence},100+sequence*.1)
    ai.update({**p,'age_s':.9,'detection_sequence':3},101.2)
    assert ai.snapshot(101.2)['confirmed_count']==0
    assert ai.tracks[0]['hits']==3
    assert ai.decision.startswith('No persistent')


def test_paused_incident_keeps_exact_frame_metadata(tmp_path):
    station = Station(Config(video_backend='aura'),tmp_path)
    try:
        station.telemetry.demo_tick((0,0,0),.1)
        first=packet()
        first['captured_at']=time.time()-100
        first['status']='paused'
        first['config']['playing']=False
        station.video.accept(first)
        jpeg=station.video.jpeg
        original_snapshot=station.telemetry.snapshot
        def advance():
            newer=packet();newer['frame']=99;newer['detections']=[];newer['captured_at']=time.time()
            station.video.accept(newer)
            return original_snapshot()
        station.telemetry.snapshot=advance
        incident=station.incident('Review paused frame')
        assert (tmp_path/incident['snapshot']).read_bytes()==jpeg
        assert incident['source_frame']==3 and incident['detections'][0]['original_label']=='pedestrian'
        assert incident['telemetry_context']=='simulated' and incident['video_context']=='video review'
        assert incident['detections_match_frame']
    finally: station.close()


@pytest.mark.parametrize('event',['pause','off','source','disconnect'])
def test_recording_stops_cleanly_when_input_changes(tmp_path,event):
    video=AuraVideo(Config(video_backend='aura'))
    first=packet();first['captured_at']=time.time()
    video.accept(first)
    store=Store(tmp_path)
    recorder=Recorder(video,store)
    try:
        recorder.start()
        deadline=time.monotonic()+3
        while not any(e['kind']=='recording_start' for e in store.events()) and time.monotonic()<deadline:
            time.sleep(.01)
        if event=='disconnect': video.disconnect('test')
        else:
            p=packet();p['captured_at']=time.time()
            if event=='pause': p['config']['playing']=False;p['status']='paused'
            if event=='off': p['config']['powered']=False
            if event=='source': p['source_epoch']='changed'
            video.accept(p)
        recorder.thread.join(3)
        assert not recorder.thread.is_alive()
        assert recorder.status.startswith('Recording stopped:'),recorder.status
        assert 'error' not in recorder.status.lower()
    finally:
        recorder.stop();store.close()


def test_native_recording_cannot_authorize_aircraft_control(tmp_path):
    station=Station(Config(mode='live',video_source='recording.mp4',live_control=True),tmp_path)
    try:
        station.video.snapshot=Mock(return_value={'age_s':.1})
        assert any('recordings cannot' in reason for reason in station.blockers())
    finally: station.close()


def test_bad_detection_boxes_are_not_forwarded():
    video=AuraVideo(Config(video_backend='aura'))
    p=packet();p['detections']=[{'label':'person','confidence':.9,'xyxy':[0,0,float('nan'),1]},
                               {'label':'person','confidence':.9,'xyxy':[10,10,0,0]}]
    video.accept(p,now=100,wall=1000)
    assert video.snapshot(now=100)['detections']==[]


def test_native_file_plays_at_recorded_rate_and_stops_at_eof(tmp_path):
    import cv2
    import numpy as np
    path=tmp_path/'short.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(64,64))
    assert writer.isOpened()
    for _ in range(3): writer.write(np.zeros((64,64,3),dtype=np.uint8))
    writer.release()
    video=Video(Config(video_source=str(path)))
    started=time.monotonic()
    video.threads[0].start()
    video.threads[0].join(4)
    try:
        assert not video.threads[0].is_alive()
        assert video.sequence==3 and video.status=='Recording ended'
        assert not video.snapshot()['simulated']
        assert not video.snapshot()['live_ready']
        assert time.monotonic()-started>=.5
    finally: video.close()


def test_api_boolean_fields_reject_truthy_strings(tmp_path):
    from fastapi.testclient import TestClient
    from groundstation.main import create_app
    config=Config()
    with TestClient(create_app(config,tmp_path)) as client:
        auth={'x-station-token':client.get('/api/session').json()['token']}
        for route,body in [('/api/lease',{'enabled':'false'}),
                           ('/api/simulation',{'gps_loss':'false'})]:
            assert client.post(route,json=body,headers=auth).status_code==422
        health=client.get('/api/health').json()
        assert health['config_fingerprint']==config_fingerprint(config)
        assert health['server_id']==client.get('/api/status').json()['server_id']
