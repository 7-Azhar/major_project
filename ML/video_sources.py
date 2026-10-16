"""Input adapters. Add future capture backends here without changing enhancement/detection."""
import os
import re
import threading

_camera_readers = {}
_camera_lock = threading.Lock()


class LatestFrameCapture:
    """Continuously drain a live capture, retaining only the newest frame.

    The reader owns release(): never release an OpenCV device from another thread
    while its driver is inside read(). A stuck driver blocks reopening that device
    until its original reader exits, instead of spawning more readers.
    """
    def __init__(self, capture, on_closed=lambda: None, timeout=5, properties=None):
        self.capture, self.on_closed, self.timeout = capture, on_closed, timeout
        self.properties = properties or {}
        self.condition = threading.Condition()
        self.stopped = self.failed = False
        self.frame = None
        self.received = self.delivered = self.skipped = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name='aura-camera')
        self.thread.start()

    def _run(self):
        try:
            while True:
                with self.condition:
                    if self.stopped:
                        break
                ok, frame = self.capture.read()
                with self.condition:
                    if not ok:
                        self.failed = True
                        self.condition.notify_all()
                        break
                    if self.stopped:
                        break
                    self.received += 1
                    self.frame = frame
                    self.condition.notify_all()
        except Exception:
            # The pipeline reports a failed read with a reconnect action.
            pass
        finally:
            try:
                self.capture.release()
            finally:
                with self.condition:
                    self.failed = True
                    self.condition.notify_all()
                self.on_closed()

    def read(self):
        with self.condition:
            self.condition.wait_for(lambda: self.stopped or self.failed or self.received > self.delivered,
                                    timeout=self.timeout)
            if self.stopped or self.received == self.delivered:
                return False, None
            self.skipped += max(0, self.received - self.delivered - 1)
            self.delivered = self.received
            return True, self.frame

    def isOpened(self):
        with self.condition:
            return not self.stopped and not self.failed

    def get(self, prop):
        # Source properties are read before constructing this wrapper.
        return self.properties.get(prop, 0)

    def release(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join(timeout=1)


def is_camera(source):
    return isinstance(source, str) and re.fullmatch(r'camera:[0-9]', source) is not None


def open_source(cv2, project, source, *, latest=False):
    if is_camera(source):
        index = int(source.split(':')[1])
        if latest:
            with _camera_lock:
                if index in _camera_readers:
                    raise RuntimeError(f'Camera {index} is still closing. Wait for the driver, then reconnect.')
                _camera_readers[index] = True
        def closed():
            with _camera_lock:
                _camera_readers.pop(index, None)
        cap = None
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if not cap.isOpened():
                raise RuntimeError(f'Camera {index} unavailable. Check connection, Windows camera permissions, and other apps using it.')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if latest:
                properties = {prop: cap.get(prop) for prop in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)}
                return LatestFrameCapture(cap, closed, properties=properties)
            return cap
        except Exception:
            if cap is not None:
                cap.release()
            if latest:
                closed()
            raise
    cap = cv2.VideoCapture(str(project / source))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Cannot open this recording. Its codec may not be supported.')
    return cap
