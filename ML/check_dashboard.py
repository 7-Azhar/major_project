"""Exercise the running local GPU dashboard API without a browser."""
import json
from pathlib import Path
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError

BASE = 'http://127.0.0.1:8765'


def get():
    with urlopen(BASE + '/api/frame', timeout=20) as response:
        return json.load(response)


def change(**values):
    request = Request(BASE + '/api/control', json.dumps(values).encode(), {'Content-Type': 'application/json'})
    with urlopen(request, timeout=20) as response:
        revision = json.load(response)['revision']
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        state = get()
        if state.get('revision', -1) >= revision:
            assert not state['error'], state
            return state
        time.sleep(.1)
    raise AssertionError('Control did not apply')


def main():
    with urlopen(BASE + '/api/options') as response:
        options = json.load(response)
    runs = []
    for enhancement in options['enhancements']:
        state = change(enhancement=enhancement, playing=False, seek=543, detection='original')
        assert state['frame'] == 543 and state['output'] and state['original']
        original = state['original']
        for detection in ('off', 'original', 'exclude25', 'exclude50'):
            state = change(detection=detection)
            assert state['frame'] == 543 and state['original'] == original
            if detection == 'off':
                assert state['detections'] == [] and state['detection_ms'] == 0
            if detection == 'exclude50':
                assert all(d['confidence'] >= .5 for d in state['detections'])
            runs.append({k: state[k] for k in ('config', 'enhancement_ms', 'detection_ms', 'excluded_count')})
        print(f"Checked {enhancement}: all four detection settings", flush=True)
    state = change(enhancement='identity', playing=True, seek=0, detection='original')
    time.sleep(.5)
    assert get()['frame'] > 0
    state = change(playing=False)
    frame = state['frame']
    time.sleep(.3)
    assert get()['frame'] == frame
    state = change(seek=state['total_frames'] - 1, playing=True)
    time.sleep(.3)
    assert get()['status'] == 'ended'
    other = options['sources'][1]
    state = change(source=other)
    assert state['osd_available'] and state['config']['detection'] == 'original'
    state = change(detection='exclude25')
    assert state['config']['detection'] == 'exclude25'
    for payload in ({'enhancement': 'invalid'}, {'seek': -1}, {'source': '../private.mp4'}, {'playing': 'yes'}):
        try:
            urlopen(Request(BASE + '/api/control', json.dumps(payload).encode(), {'Content-Type': 'application/json'}))
            raise AssertionError('Invalid control accepted')
        except HTTPError as error:
            assert error.code == 400
    final = change(source=options['sources'][0], enhancement='identity', detection='exclude25', seek=543, playing=False)
    assert final['frame'] == 543
    output = Path(__file__).resolve().parent / 'outputs/dashboard_checks.json'
    output.write_text(json.dumps(dict(passed=True, combinations=runs,
        checks=['32 enhancement/detection combinations', 'pause freezes frame', 'play advances', 'seek',
                'end of recording', 'exclusion available on other sources', 'invalid controls rejected']), indent=2))
    print(f'All checks passed: {output}')


if __name__ == '__main__':
    main()
