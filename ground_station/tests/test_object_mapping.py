"""Known geometry, missing hardware and duplicate-object regression cases."""
from dataclasses import asdict
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pymavlink import mavutil

from groundstation.config import Config
from groundstation.geolocation import Calibration, locate, rotation
from groundstation.main import create_app
from groundstation.observations import Observations, deduplicate
from groundstation.storage import Store
from groundstation.station import Station
from groundstation.telemetry import Telemetry

BOX=[.45,.3,.55,.5]


def calibrated(**kwargs):
    return Calibration(verified=True,mount_pitch_down_deg=45,
        ground_elevation_msl_m=80,ground_reference_latitude=17,ground_reference_longitude=78,
        gps_error_m=1,height_error_m=.5,angle_error_deg=.5,pixel_error_px=1,**kwargs)


def pose(**kwargs):
    return dict(latitude=17,longitude=78,altitude_msl_m=100,roll_deg=0,pitch_deg=0,yaw_deg=0,
                speed_mps=0,pose_skew_ms=0,**kwargs)


def test_known_ray_is_twenty_metres_north_not_aircraft_gps():
    result=locate(BOX,pose(),calibrated(),[640,480])
    assert math.radians(result['latitude']-17)*6378137 == pytest.approx(20,abs=.01)
    assert result['longitude']==pytest.approx(78)
    assert result['estimated'] and result['error_radius_m']>=1
    assert result['range_m']==20


def test_heading_turns_north_ray_east():
    p=pose();p['yaw_deg']=90
    result=locate(BOX,p,calibrated(),[640,480])
    assert result['latitude']==pytest.approx(17)
    assert math.radians(result['longitude']-78)*6378137*math.cos(math.radians(17))==pytest.approx(20,abs=.01)


def test_nadir_centre_is_under_aircraft_and_roll_rotates_ray():
    c=calibrated();c.mount_pitch_down_deg=90
    result=locate(BOX,pose(),c,[1280,960])
    assert result['range_m']==0
    p=pose();p['roll_deg']=10
    rolled=locate(BOX,p,c,[640,480])
    assert rolled['longitude']<78
    assert (rotation(10,20,30).T@rotation(10,20,30)).diagonal()==pytest.approx([1,1,1])


@pytest.mark.parametrize('case',['unverified','aspect','clipped','horizon','height','missing_pose','survey','uncertainty','range'])
def test_unreliable_geometry_never_produces_a_coordinate(case):
    c=calibrated();p=pose();box=BOX;size=[640,480]
    if case=='unverified':c.verified=False
    if case=='aspect':size=[1920,1080]
    if case=='clipped':box=[.4,.3,.6,1]
    if case=='horizon':c.mount_pitch_down_deg=0
    if case=='height':p['altitude_msl_m']=80
    if case=='missing_pose':p.pop('yaw_deg')
    if case=='survey':c.ground_reference_latitude=18
    if case=='uncertainty':c.max_error_m=1
    if case=='range':c.max_range_m=5
    with pytest.raises(ValueError):locate(box,p,c,size)


@pytest.mark.parametrize('data',[{'verified':'false'},{'verified':True},{'fx':float('nan')},
                                {'source':'recording.mp4'},{'unknown':0},{'distortion':[0,0]}])
def test_calibration_rejects_incomplete_or_ambiguous_values(data):
    with pytest.raises(ValueError):Calibration.parse(data)


def frame(sequence=1, detections=None, **kwargs):
    return dict(source_id='camera-session-1',source='camera:1',age_s=.1,
        detection_sequence=sequence,source_frame=sequence,captured_monotonic=100,
        image_size=[640,480],live_ready=True,
        detections=detections if detections is not None else [{'label':'person','confidence':.9,'box':BOX}],**kwargs)


@pytest.fixture
def tracking(tmp_path):
    store=Store(tmp_path)
    tracker=Observations(store);tracker.calibration=calibrated()
    telemetry=SimpleNamespace(pose_epoch=0,pose_at=Mock(return_value=pose()))
    yield tracker,telemetry
    store.close()


def confirm(tracker,telemetry,first=1, detections=None):
    for seq in range(first,first+3):tracker.update(frame(seq,detections),telemetry,100+seq*.1)
    return tracker.snapshot(100+(first+2)*.1)['items']


def test_same_frame_duplicate_boxes_and_repeat_poll_only_count_once(tracking):
    tracker,t=tracking
    doubles=[{'label':'person','confidence':.9,'box':BOX},{'label':'people','confidence':.8,'box':BOX}]
    tracker.update(frame(1,doubles),t,100.1)
    for _ in range(8):tracker.update(frame(1,doubles),t,100.2)
    item=tracker.snapshot(100.2)['items']
    assert len(item)==1 and item[0]['hits']==1 and not item[0]['confirmed'] and item[0]['location'] is None
    tracker.update(frame(2,doubles),t,100.3);tracker.update(frame(3,doubles),t,100.4)
    assert tracker.snapshot(100.4)['mapped_count']==1


def test_brief_occlusion_reuses_track_and_preserves_review(tracking):
    tracker,t=tracking
    first=confirm(tracker,t)[0]
    tracker.review(first['id'],'verified')
    tracker.update(frame(4,[]),t,100.4)
    assert tracker.snapshot(100.4)['items'][0]['status']=='last seen'
    tracker.update(frame(5),t,100.5)
    again=tracker.snapshot(100.5)['items']
    assert len(again)==1 and again[0]['id']==first['id'] and again[0]['review']=='verified'
    assert len([e for e in tracker.store.events() if e['kind']=='object_confirmed'])==1


def test_two_people_and_vehicle_stay_distinct(tracking):
    tracker,t=tracking
    items=confirm(tracker,t,detections=[{'label':'person','confidence':.9,'box':BOX},
        {'label':'person','confidence':.9,'box':[.6,.3,.7,.5]},
        {'label':'car','confidence':.9,'box':[.2,.3,.35,.5]}])
    assert len(items)==3 and len({x['id'] for x in items})==3
    assert [x['category'] for x in items].count('vehicle')==1


def test_source_change_starts_new_track_and_stale_frame_never_counts(tracking):
    tracker,t=tracking
    old=confirm(tracker,t)[0]['id']
    video=frame(4);video['source_id']='camera-session-2'
    tracker.update(video,t,100.5)
    item=tracker.snapshot(100.5)['items'][0]
    assert item['id']!=old and item['hits']==1
    video['age_s']=None
    tracker.update(video,t,102)
    assert not tracker.snapshot(102)['items'][0]['visible']


@pytest.mark.parametrize('reason',['recorded','missing_pose','wrong_camera','uncalibrated'])
def test_detected_objects_remain_unlocated_without_valid_evidence(tracking,reason):
    tracker,t=tracking
    if reason=='missing_pose':t.pose_at.side_effect=ValueError('No aligned telemetry')
    if reason=='wrong_camera':tracker.calibration.source='camera:2'
    if reason=='uncalibrated':tracker.calibration.verified=False
    for seq in (1,2,3):
        v=frame(seq)
        if reason=='recorded':v['live_ready']=False
        tracker.update(v,t,100+seq*.1)
    snap=tracker.snapshot(100.3)
    assert snap['confirmed_count']==1 and snap['mapped_count']==0
    assert snap['items'][0]['location'] is None


def test_dismissed_track_is_not_counted_or_realerted(tracking):
    tracker,t=tracking
    item=confirm(tracker,t)[0];tracker.review(item['id'],'dismissed')
    tracker.update(frame(4),t,100.4)
    assert tracker.snapshot(100.4)['confirmed_count']==0
    assert tracker.snapshot(100.4)['items'][0]['id']==item['id']


def test_ambiguous_crossing_starts_candidate_instead_of_merging_people(tracking):
    tracker,t=tracking
    tracker.calibration.verified=False
    people=[{'label':'person','confidence':.9,'box':[.3,.3,.5,.6]},
            {'label':'person','confidence':.9,'box':[.5,.3,.7,.6]}]
    old={item['id'] for item in confirm(tracker,t,detections=people)}
    tracker.update(frame(4,[{'label':'person','confidence':.9,'box':[.4,.3,.6,.6]}]),t,100.4)
    items=tracker.snapshot(100.4)['items']
    current=[item for item in items if item['visible']]
    assert len(current)==1 and current[0]['id'] not in old
    assert current[0]['association']=='ambiguous; new track'
    assert not current[0]['confirmed']
    assert all(item['status']=='last seen' for item in items if item['id'] in old)


def test_fleeting_candidates_expire_before_confirmed_observations(tracking):
    tracker,t=tracking
    old=confirm(tracker,t)[0]['id']
    tracker.update(frame(4,[{'label':'car','confidence':.8,'box':[.7,.3,.8,.5]}]),t,100.4)
    tracker.update(frame(5,[]),t,104)
    assert [item['id'] for item in tracker.snapshot(104)['items']]==[old]
    tracker.update(frame(6,[]),t,281)
    assert tracker.snapshot(281)['items']==[]


def test_telemetry_epoch_change_clears_old_location_and_review(tracking):
    tracker,t=tracking
    previous=confirm(tracker,t)[0]
    tracker.review(previous['id'],'verified')
    t.pose_epoch+=1
    tracker.update(frame(4),t,100.4)
    current=tracker.snapshot(100.4)['items']
    assert len(current)==1 and current[0]['id']!=previous['id']
    assert current[0]['review']=='unreviewed' and current[0]['location'] is None


@pytest.mark.parametrize('captured_sequence,source_id,expected',[(8,'0',1),(9,'0',0),(8,'old-camera-session',0)])
def test_incident_only_attaches_locations_from_the_captured_frame(tmp_path,captured_sequence,source_id,expected):
    station=Station(Config(video_source=0),tmp_path)
    try:
        station.video.sequence=captured_sequence
        station.video.detection_sequence=8
        item=dict(confirmed=True,location={'latitude':17,'longitude':78},review='unreviewed',
                  frame=8,source='0',source_id=source_id)
        station.observations.snapshot=Mock(return_value={'items':[item]})
        incident=station.incident('Software test of frame correlation')
        assert len(incident['object_locations'])==expected
    finally: station.close()


def test_mavlink_pose_uses_independent_time_aligned_samples(monkeypatch):
    telemetry=Telemetry(Config(mode='live',video_source=0))
    clock=[100.]
    monkeypatch.setattr('groundstation.telemetry.time.monotonic',lambda:clock[0])
    tx=mavutil.mavlink.MAVLink(None,srcSystem=1,srcComponent=1)
    rx=mavutil.mavlink.MAVLink(None)
    messages=[mavutil.mavlink.MAVLink_global_position_int_message(1000,170000000,780000000,100000,10000,100,0,0,0),
              mavutil.mavlink.MAVLink_attitude_message(1000,0,0,0,0,0,0),
              mavutil.mavlink.MAVLink_gps_raw_int_message(0,3,170000000,780000000,100000,100,100,0,0,14)]
    for msg in messages:telemetry.accept(rx.parse_char(msg.pack(tx)))
    c=calibrated();c.video_delay_ms=200;c.telemetry_delay_ms=100
    p=telemetry.pose_at(100.1,c)
    assert p['altitude_msl_m']==100 and p['latitude']==17 and p['pose_skew_ms']==0
    with pytest.raises(ValueError,match='time'):telemetry.pose_at(102,c)
    telemetry.history['attitude'].clear()
    with pytest.raises(ValueError,match='attitude'):telemetry.pose_at(100.1,c)
    with pytest.raises(ValueError,match='Real MAVLink'):Telemetry(Config()).pose_at(100,c)


def test_mapping_routes_and_calibration_persist_without_enabling_control(tmp_path):
    config=Config()
    with TestClient(create_app(config,tmp_path)) as client:
        auth={'x-station-token':client.get('/api/session').json()['token']}
        assert 'Detection map' in client.get('/').text
        assert client.get('/advanced').status_code==200
        data=client.get('/api/mapping/calibration').json()['calibration']
        assert not data['verified']
        assert client.post('/api/mapping/calibration',json=data).status_code==403
        assert client.post('/api/mapping/calibration',json={'verified':True},headers=auth).status_code==422
        assert client.post('/api/mapping/calibration',json=data,headers=auth).status_code==200
        assert not client.get('/api/system').json()['live_control']
        assert client.get('/api/observations/geojson').json()['features']==[]
        assert client.get('/api/observations/export').status_code==200
        example=client.get('/api/observations/example').json()
        assert example['simulated'] and len(example['items'])==2
        assert all(t['location']['estimated'] for t in example['items'])
        assert client.get('/api/observations/geojson').json()['features']==[]
        assert client.get('/mapping-guide').status_code==200
        assert client.post('/api/observations/missing/review',json={'status':'verified'},headers=auth).status_code==404
    store=Store(tmp_path)
    try: assert asdict(Observations(store).calibration)==data
    finally: store.close()
