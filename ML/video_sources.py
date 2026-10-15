"""Input adapters. Add future capture backends here without changing enhancement/detection."""
import os
import re


def is_camera(source):
    return isinstance(source, str) and re.fullmatch(r'camera:[0-9]', source) is not None


def open_source(cv2, project, source):
    if is_camera(source):
        index = int(source.split(':')[1])
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f'Camera {index} unavailable. Check connection, Windows camera permissions, and other apps using it.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    cap = cv2.VideoCapture(str(project / source))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Cannot open this recording. Its codec may not be supported.')
    return cap
