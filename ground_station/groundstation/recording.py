import threading
import time
import cv2
from .video_evidence import capture_evidence


class Recorder:
    """Independent bounded recording; capture never waits for disk."""
    def __init__(self, video, store):
        self.video, self.store = video, store
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.filename = None
        self.status = "Not recording"
        self.started_at = 0

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("Already recording")
            evidence = capture_evidence(self.video, copy_frame=True)
            age = evidence['state']['age_s']
            if evidence['frame'] is None or age is None or age > 1:
                raise ValueError("Fresh video required")
            self.filename = f"recording-{time.time_ns()}.avi"
            self.started_at = time.monotonic()
            self.stop_event.clear()
            self.status = "Recording at 10 fps (maximum 5 minutes / 500 MB)"
            self.thread = threading.Thread(target=self.run, args=(evidence,), daemon=True)
            self.thread.start()
            return {"filename": self.filename, "status": self.status}

    def run(self, first):
        writer = None
        filename = self.filename
        path = self.store.root / filename
        try:
            frame = first['frame']
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
            if not writer.isOpened():
                raise RuntimeError("MJPG AVI encoder unavailable")
            self.store.add("recording_start", {"filename": filename, "fps": 10,
                "source_id":first['source_id'], "source":first['state'].get('source')})
            while not self.stop_event.is_set():
                tick = time.monotonic()
                if tick-self.started_at >= 300 or (path.exists() and path.stat().st_size >= 500_000_000):
                    self.status = "Recording stopped at storage/time limit"
                    break
                evidence = capture_evidence(self.video, copy_frame=True)
                frame, age = evidence['frame'], evidence['state']['age_s']
                if evidence['source_id'] != first['source_id']:
                    self.status = 'Recording stopped: input changed or recording was seeked'
                    break
                if frame is None or age is None or age > 1:
                    self.status = "Recording stopped: video paused, stale or unavailable"
                    break
                if frame.shape[:2] != (h, w):
                    self.status = 'Recording stopped: video dimensions changed'
                    break
                writer.write(frame)
                self.stop_event.wait(max(0, .1-(time.monotonic()-tick)))
        except Exception as exc:
            self.status = f"Recording error: {exc}"
        finally:
            if writer:
                writer.release()
            if self.status.startswith("Recording at"):
                self.status = "Recording saved"
            self.store.add("recording_stop", {"filename": filename, "status": self.status})

    def snapshot(self):
        return {"active": bool(self.thread and self.thread.is_alive()), "status": self.status,
                "filename": self.filename}

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(3)
        return self.snapshot()
