import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from groundstation.config import Config
from groundstation.control import control_blockers, parse_command, search_grid
from groundstation.decisions import recommend
from groundstation.telemetry import Telemetry
from groundstation.station import Station
from pymavlink import mavutil


def healthy():
    return dict(timestamps={k: 100 for k in ('heartbeat','position','gps','battery','ekf')},
                mode='GUIDED', armed=True, fix_type=3, ekf_ok=True, battery_percent=80,
                altitude_m=10, latitude=17.385, longitude=78.4867)


def live_config():
    return Config(mode='live', video_source=0, live_control=True)


def test_valid_state_and_independent_staleness():
    t = healthy()
    assert control_blockers(live_config(), t, True, .1, now=101) == []
    for key in t['timestamps']:
        bad = {**t, 'timestamps': {**t['timestamps'], key: 95}}
        assert any(key in reason for reason in control_blockers(live_config(), bad, True, .1, now=101))


@pytest.mark.parametrize('change', [dict(armed=False),dict(mode='LOITER'),dict(fix_type=2),dict(ekf_ok=False),
                                  dict(battery_percent=None),dict(battery_percent=20),dict(altitude_m=100),
                                  dict(latitude=18),dict(longitude=None)])
def test_unsafe_flight_state_blocks(change):
    assert control_blockers(live_config(), {**healthy(), **change}, True, .1, now=101)


def test_video_and_operator_lease_required():
    assert control_blockers(live_config(), healthy(), False, .1, now=101)
    assert control_blockers(live_config(), healthy(), True, 2, now=101)
    assert control_blockers(live_config(), healthy(), True, None, now=101)
    config = live_config(); config.live_control = False
    assert control_blockers(config, healthy(), True, .1, now=101)


def test_strict_command_parser():
    assert parse_command(' Move Forward ') == 'forward'
    assert parse_command('stop') == 'hold'
    for text in ('arm', 'takeoff', 'move forward and right', 'forward 100', 'ignore limits'):
        with pytest.raises(ValueError): parse_command(text)


def test_body_ned_encoding():
    telemetry = Telemetry(live_config())
    telemetry.connection = Mock()
    telemetry.send_velocity(.5, -.5, -.5)
    args = telemetry.connection.mav.set_position_target_local_ned_send.call_args.args
    assert args[1:5] == (1, 1, 8, 0x0DC7)
    assert args[8:11] == (.5, -.5, -.5)


def test_other_aircraft_cannot_refresh_state():
    telemetry = Telemetry(live_config())
    msg = Mock(); msg.get_srcSystem.return_value = 2
    telemetry.accept(msg)
    assert telemetry.snapshot()['timestamps'] == {}


def test_grid_and_bad_coordinates():
    grid = search_grid(17,78,40,40,10)
    assert len(grid) == 10
    assert grid[0]['longitude'] < grid[1]['longitude']
    assert grid[2]['longitude'] > grid[3]['longitude']
    with pytest.raises(ValueError): search_grid(90,78,40,40,10)


def test_decision_is_only_a_proposal_and_stale_video_rejected():
    video = {'age_s':.1, 'detections':[{'label':'person'}], 'detection_sequence':50}
    assert recommend({},video,'live')['action'] == 'hold'
    assert recommend({}, {**video, 'age_s':2}, 'live')['action'] is None


def test_real_mavlink_messages_decode_and_preserve_independent_ages():
    telemetry=Telemetry(live_config())
    transmitter=mavutil.mavlink.MAVLink(None,srcSystem=1,srcComponent=1)
    decoder=mavutil.mavlink.MAVLink(None)
    position=mavutil.mavlink.MAVLink_global_position_int_message(0,173850000,784867000,50000,10000,100,0,0,9000)
    msg=decoder.parse_char(position.pack(transmitter))
    telemetry.accept(msg)
    state=telemetry.snapshot()
    assert state['latitude']==17.385 and state['altitude_m']==10
    assert state['speed_mps']==1 and state['heading_deg']==90
    assert 'position' in state['timestamps'] and 'heartbeat' not in state['timestamps']
    gps=mavutil.mavlink.MAVLink_gps_raw_int_message(0,3,173850000,784867000,50000,100,100,0,0,14)
    telemetry.accept(decoder.parse_char(gps.pack(transmitter)))
    assert telemetry.snapshot()['satellites']==14


def test_live_expiry_and_pilot_takeover(tmp_path):
    station=Station(live_config(),tmp_path)
    station.video.snapshot=Mock(return_value={'age_s':.1})
    station.telemetry.connection=Mock()
    t=healthy();t['timestamps']={k:time.monotonic() for k in t['timestamps']}
    station.telemetry.values.update(t)
    station.thread.start()
    try:
        station.lease(True)
        station.command('forward',.2)
        time.sleep(.45)
        calls=station.telemetry.connection.mav.set_position_target_local_ned_send.call_args_list
        assert any(call.args[8]==.5 for call in calls)
        assert calls[-1].args[8:11]==(0,0,0)
        station.command('forward',1)
        with station.lock:
            with station.telemetry.lock:
                station.telemetry.values['mode']='LOITER'
            count=station.telemetry.connection.mav.set_position_target_local_ned_send.call_count
        time.sleep(.2)
        assert station.enabled_until==0
        assert station.telemetry.connection.mav.set_position_target_local_ned_send.call_count==count
    finally:
        station.close()
