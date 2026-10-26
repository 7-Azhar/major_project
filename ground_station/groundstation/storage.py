import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "station.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, time TEXT, kind TEXT, payload TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS missions (id INTEGER PRIMARY KEY, name TEXT, plan TEXT)")
        self.db.commit()

    def add(self, kind, payload):
        with self.lock:
            cursor = self.db.execute("INSERT INTO events(time,kind,payload) VALUES(?,?,?)",
                                     (datetime.now(timezone.utc).isoformat(), kind, json.dumps(payload)))
            self.db.commit()
            return cursor.lastrowid

    def events(self, limit=100, include_telemetry=True):
        with self.lock:
            clause = "" if include_telemetry else " WHERE kind != 'telemetry'"
            rows = self.db.execute("SELECT id,time,kind,payload FROM events" + clause + " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "time": r[1], "kind": r[2], "payload": json.loads(r[3])} for r in rows]

    def close(self):
        with self.lock:
            self.db.close()

    def save_mission(self,name,points):
        with self.lock:
            cursor = self.db.execute('INSERT INTO missions(name,plan) VALUES(?,?)',(name,json.dumps(points)))
            self.db.commit()
            return cursor.lastrowid

    def missions(self):
        with self.lock:
            rows = self.db.execute('SELECT id,name,plan FROM missions ORDER BY id DESC LIMIT 100').fetchall()
        return [{'id':r[0],'name':r[1],'points':json.loads(r[2])} for r in rows]
