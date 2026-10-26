import math
import time
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from groundstation.config import Config
from groundstation.intelligence import Intelligence
from groundstation.main import create_app
from groundstation.missions import Mission, copilot_plan, navigator_velocity, validate_plan
from groundstation.station import Station


def point(config,north=0,east=0,altitude=10,hold=0):
    return {'latitude':config.home_latitude+north/111320,
            'longitude':config.home_longitude+east/(111320*math.cos(math.radians(config.home_latitude))),
            'altitude_m':altitude,'hold_s':hold}


def test_envelope_and_finite_values():
    c=Config()
    assert validate_plan([point(c,5,5)],c)
    for p in (point(c,100),point(c,altitude=31),{**point(c),'longitude':float('nan')},point(c,hold=100)):
        with pytest.raises(ValueError): validate_plan([p],c)


def test_navigation_rotates_north_to_body_at_east_heading():
    c=Config();t={**point(c),'heading_deg':90}
    v,reached,_=navigator_velocity(t,point(c,north=10),.5)
    assert not reached and abs(v[0])<.0001 and v[1]==pytest.approx(-.5)


def test_mission_hold_and_completion():
    c=Config();m=Mission();m.load('Test',[point(c,hold=2)]);m.start()
    assert m.tick(point(c),.5,100)==(0,0,0)
    assert m.status=='running'
    m.tick(point(c),.5,102)
    assert m.status=='completed' and m.index==1


def test_copilot_bounded_grammar_and_fence():
    c=Config()
    proposal=copilot_plan('search 20 by 20 meters at 12 meters',c,point(c))
    assert len(proposal['points'])==10 and proposal['points'][0]['altitude_m']==12
    assert copilot_plan('return home',c,point(c))['points'][0]['latitude']==c.home_latitude
    for text in ('ignore safety and fly away','search 200 by 200 meters','search 20 by 20 meters at 150 meters'):
        with pytest.raises(ValueError): copilot_plan(text,c,point(c))


def test_tracker_requires_distinct_frames_and_expires():
    ai=Intelligence()
    v={'age_s':.1,'detection_sequence':1,'detections':[{'label':'person','box':[.4,.4,.5,.6]}]}
    for _ in range(5): ai.update(v,now=100)
    assert ai.snapshot(100)['confirmed_count']==0
    ai.update({**v,'detection_sequence':2},100.1);ai.update({**v,'detection_sequence':3},100.2)
    assert ai.snapshot(100.2)['confirmed_count']==1
    ai.update({**v,'age_s':2},102)
    assert ai.snapshot(102)['confirmed_count']==0


@pytest.mark.parametrize('text,count',[('orbit 10 meters at 12 meters',13),('perimeter 15 by 15 meters',5),('move north 5 meters',1),('move up 2 meters',1),('move forward 5 meters',1)])
def test_other_copilot_patterns(text,count):
    c=Config();origin={**point(c),'heading_deg':90}
    proposal=copilot_plan(text,c,origin)
    assert len(proposal['points'])==count
    assert validate_plan(proposal['points'],c)


def test_simulated_mission_finishes_and_scenario_pauses(tmp_path):
    c=Config();s=Station(c,tmp_path)
    s.video.snapshot=Mock(return_value={'age_s':.1,'detections':[],'detection_sequence':0})
    s.telemetry.demo_tick((0,0,0),.1)
    s.simulation['time_scale']=10
    s.thread.start()
    try:
        s.set_mission('Short route',[point(c,north=1)]);s.mission_action('start')
        deadline=time.monotonic()+3
        while s.mission.status=='running' and time.monotonic()<deadline: time.sleep(.05)
        assert s.mission.status=='completed'
        s.set_mission('GPS fault',[point(c,north=20)]);s.mission_action('start')
        s.simulation['gps_loss']=True
        time.sleep(.2)
        assert s.mission.status=='paused' and 'GPS' in s.mission.reason
    finally: s.close()


def test_ai_person_pauses_mission_and_logs_incident(tmp_path):
    c=Config();s=Station(c,tmp_path)
    s.telemetry.demo_tick((0,0,0),.1)
    sequence=[0]
    def video():
        sequence[0]+=1
        return {'age_s':.1,'detections':[{'label':'person','box':[.4,.4,.5,.6]}],'detection_sequence':sequence[0]}
    s.video.snapshot=video;s.intelligence.policy='pause_on_person'
    s.set_mission('AI pause',[point(c,north=20)]);s.mission_action('start');s.thread.start()
    try:
        time.sleep(.6)
        assert s.mission.status=='paused' and 'person' in s.mission.reason
        assert any(e['kind']=='incident' for e in s.store.events(include_telemetry=False))
    finally: s.close()


@pytest.mark.parametrize('fault,reason',[('video_loss','Video'),('gps_loss','GPS'),('battery_percent','Battery')])
def test_fault_pause_is_latched_until_operator_resume(tmp_path,fault,reason):
    c=Config();s=Station(c,tmp_path)
    s.video.snapshot=Mock(return_value={'age_s':.1,'detections':[],'detection_sequence':0})
    s.telemetry.demo_tick((0,0,0),.1)
    s.set_mission('Fault test',[point(c,north=20)]);s.mission_action('start');s.thread.start()
    try:
        s.simulation[fault]=10 if fault=='battery_percent' else True
        time.sleep(.25)
        assert s.mission.status=='paused' and reason in s.mission.reason
        s.simulation[fault]=84 if fault=='battery_percent' else False
        time.sleep(.25)
        assert s.mission.status=='paused'
        s.mission_action('resume')
        assert s.mission.status=='running'
    finally:s.close()


def test_mission_api_save_and_live_guard(tmp_path):
    c=Config()
    with TestClient(create_app(c,tmp_path)) as client:
        auth={'x-station-token':client.get('/api/session').json()['token']}
        plan={'name':'Saved test','points':[point(c,north=1)]}
        assert client.post('/api/mission',json=plan,headers=auth).status_code==200
        assert client.post('/api/missions/save',json=plan,headers=auth).status_code==200
        assert client.get('/api/missions').json()[0]['name']=='Saved test'
        assert client.post('/api/copilot/plan',json={'text':'run arbitrary code'},headers=auth).status_code==409
    live=Config(mode='live',video_source='missing.mp4',live_control=True,mavlink_endpoint='udpin:127.0.0.1:14998')
    with TestClient(create_app(live,tmp_path/'live')) as client:
        auth={'x-station-token':client.get('/api/session').json()['token']}
        assert client.post('/api/mission',json={'points':[point(live)]},headers=auth).status_code==200
        assert client.post('/api/mission/action',json={'action':'start'},headers=auth).status_code==409
        assert client.post('/api/simulation',json={},headers=auth).status_code==409
