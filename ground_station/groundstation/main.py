import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictBool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Config, ROOT
from .control import parse_command, search_grid
from .station import Station
from typing import Literal
from serial.tools import list_ports


class Command(BaseModel):
    text: str = Field(min_length=1, max_length=100)
    duration_s: float = Field(default=0.5, ge=.1, le=1)


class Lease(BaseModel):
    enabled: StrictBool


class Incident(BaseModel):
    note: str = Field(min_length=1, max_length=2000)


class Grid(BaseModel):
    latitude: float
    longitude: float
    width_m: float = 40
    height_m: float = 40
    spacing_m: float = 10


class WaypointInput(BaseModel):
    latitude: float = Field(ge=-85,le=85,allow_inf_nan=False)
    longitude: float = Field(ge=-180,le=180,allow_inf_nan=False)
    altitude_m: float = Field(default=10,ge=1,le=120,allow_inf_nan=False)
    hold_s: float = Field(default=0,ge=0,le=60,allow_inf_nan=False)


class MissionInput(BaseModel):
    name: str = Field(default='Untitled search',min_length=1,max_length=100)
    points: list[WaypointInput] = Field(min_length=1,max_length=100)


class MissionAction(BaseModel):
    action: Literal['start','pause','resume','cancel']


class CopilotInput(BaseModel):
    text: str = Field(min_length=1,max_length=300)


class PolicyInput(BaseModel):
    policy: Literal['advisory','pause_on_person']


class ReviewInput(BaseModel):
    status: Literal['unreviewed','verified','dismissed']


class SimulationInput(BaseModel):
    person_visible: StrictBool = False
    gps_loss: StrictBool = False
    video_loss: StrictBool = False
    battery_percent: int = Field(default=84,ge=0,le=100)
    time_scale: int = Field(default=1,ge=1,le=10)


def create_app(config=None, data_root=None):
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        app.state.station = Station(config or Config.load(), data_root)
        try:
            app.state.station.start()
            yield
        finally:
            app.state.station.close()

    app = FastAPI(title="Aero Ground Station", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def protect(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Same-origin requests only"}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if not secrets.compare_digest(request.headers.get("x-station-token", ""), token):
                return JSONResponse({"detail": "Station token required"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: https://services.arcgisonline.com https://tile.openstreetmap.org; connect-src 'self'; frame-src 'self'; frame-ancestors 'self'"
        return response

    def station():
        return app.state.station

    from .vision_routes import install_vision_routes
    install_vision_routes(app, token)

    @app.get('/api/health')
    def health():
        from .config import config_fingerprint
        return {'application': 'AURA Ground Station', 'video_backend': station().config.video_backend,
                'aura_port': station().config.aura_port, 'server_id':station().server_id,
                'config_fingerprint':config_fingerprint(station().config)}

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web" / "overview.html")

    @app.get("/advanced")
    def advanced():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get('/mapping-guide')
    def mapping_guide():
        return FileResponse(ROOT / 'docs' / 'TARGET_MAPPING.md',media_type='text/plain; charset=utf-8')

    @app.get('/api/mapping/calibration')
    def mapping_calibration():
        with station().lock:
            return station().observations.calibration_status()

    @app.post('/api/mapping/calibration')
    def save_mapping_calibration(body: dict):
        try:
            with station().lock:
                return station().observations.set_calibration(body)
        except (ValueError,TypeError) as exc:
            raise HTTPException(422,str(exc)) from exc

    @app.post('/api/observations/{key}/review')
    def review_observation(key: str, body: ReviewInput):
        try:
            with station().lock:
                return station().observations.review(key,body.status)
        except ValueError as exc:
            raise HTTPException(404,str(exc)) from exc

    @app.get('/api/observations/export')
    def export_observations():
        with station().lock:
            data=station().observations.snapshot()
        return JSONResponse(data,headers={'Content-Disposition':'attachment; filename="aura-observations.json"'})

    @app.get('/api/observations/example')
    def mapping_example():
        # Isolated illustration. Never enters live tracking, calibration, logs or control.
        from .geolocation import Calibration, locate
        c=Calibration(verified=True,mount_pitch_down_deg=60,ground_elevation_msl_m=80,
                      ground_reference_latitude=17,ground_reference_longitude=78,
                      gps_error_m=2,height_error_m=1,angle_error_deg=1,pixel_error_px=2)
        pose=dict(latitude=17,longitude=78,altitude_msl_m=100,roll_deg=0,pitch_deg=0,yaw_deg=30)
        items=[]
        for number,category,box in [(1,'person',[.25,.3,.35,.55]),(2,'vehicle',[.6,.3,.8,.6])]:
            items.append(dict(id='sample-'+str(number),title='Sample '+category,category=category,label=category,
                              box=box,confidence=.9,confirmed=True,visible=True,age_s=0,status='sample',
                              review='synthetic example',location=locate(box,pose,c,[640,480]),location_reason='',
                              simulated=True))
        return {'simulated':True,'items':items,'aircraft':pose,'reason':'Synthetic illustration · unrelated to the video feed'}

    @app.get('/api/observations/geojson')
    def export_locations():
        with station().lock:
            items=station().observations.snapshot()['items']
        return JSONResponse({'type':'FeatureCollection','features':[
            {'type':'Feature','geometry':{'type':'Point','coordinates':[t['location']['longitude'],t['location']['latitude']]},
             'properties':{k:t[k] for k in ('id','title','label','confidence','review','status','last_seen_utc','location')}}
            for t in items if t['confirmed'] and t['location'] and t['review']!='dismissed']},
            headers={'Content-Disposition':'attachment; filename="aura-estimated-locations.geojson"'})

    @app.get("/docs")
    def docs():
        return FileResponse(ROOT / "web" / "api.html")

    @app.post("/api/recording/start")
    def start_recording():
        try:
            return station().recorder.start()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/recording/stop")
    def stop_recording():
        return station().recorder.stop()

    @app.get("/recordings/{filename}")
    def download_recording(filename: str):
        if not filename.startswith("recording-") or not filename.endswith(".avi") or "/" in filename or "\\" in filename:
            raise HTTPException(404)
        if station().recorder.snapshot()["active"] and station().recorder.filename == filename:
            raise HTTPException(409, "Stop recording before downloading")
        path = station().store.root / filename
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, filename=filename, media_type="video/x-msvideo")

    @app.get("/api/session")
    def session():
        return {"token": token}

    @app.get("/api/status")
    def status():
        return station().state()

    @app.post('/api/mission')
    def plan(body: MissionInput):
        try:
            return station().set_mission(body.name,[p.model_dump() for p in body.points])
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from exc

    @app.post('/api/mission/action')
    def mission_action(body: MissionAction):
        try:
            return station().mission_action(body.action)
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from exc

    @app.get('/api/missions')
    def saved_missions():
        return station().store.missions()

    @app.post('/api/missions/save')
    def save_mission(body: MissionInput):
        try:
            from .missions import validate_plan
            points=validate_plan([p.model_dump() for p in body.points],station().config)
            return {'id':station().store.save_mission(body.name,points),'name':body.name}
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from exc

    @app.post('/api/copilot/plan')
    def copilot(body: CopilotInput):
        try:
            return station().copilot(body.text)
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from exc

    @app.post('/api/ai/policy')
    def ai_policy(body: PolicyInput):
        with station().lock:
            station().intelligence.policy=body.policy
            station().store.add('ai_policy',{'policy':body.policy})
        return {'policy':body.policy}

    @app.post('/api/simulation')
    def simulation(body: SimulationInput):
        s=station()
        if s.config.mode!='demo':
            raise HTTPException(409,'Simulation controls are unavailable in live mode')
        if s.config.video_backend == 'aura' and body.person_visible:
            raise HTTPException(409, 'AURA uses real video inference; person injection requires the native demo backend')
        with s.lock:
            s.simulation=body.model_dump()
            s.video.demo_person_visible=body.person_visible
            s.video.demo_video_loss=body.video_loss
        return s.simulation

    @app.get('/api/system')
    def system():
        s=station()
        return {'mode':s.config.mode,'video_source':s.video.snapshot()['source'],
                'video_backend':s.config.video_backend,'aura_port':s.config.aura_port,
                'mavlink_endpoint':s.config.mavlink_endpoint,'mavlink_baud':s.config.mavlink_baud,
                'target_system':s.config.target_system,'target_component':s.config.target_component,
                'live_control':s.config.live_control,'live_missions':s.config.live_missions,
                'serial_ports':[{'port':p.device,'description':p.description} for p in list_ports.comports()]}

    @app.get("/api/stream")
    async def stream(request: Request):
        async def generate():
            while not await request.is_disconnected():
                yield "data: " + json.dumps(station().state()) + "\n\n"
                await asyncio.sleep(.25)
        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/video.mjpg")
    async def video(request: Request):
        async def generate():
            previous = -1
            while not await request.is_disconnected():
                v = station().video
                with v.lock:
                    jpeg, seq = v.jpeg, v.sequence
                if jpeg and seq != previous:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                    previous = seq
                await asyncio.sleep(.015)
        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/command")
    def command(body: Command):
        try:
            return station().command(parse_command(body.text), body.duration_s)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/lease")
    def lease(body: Lease):
        try:
            return station().lease(body.enabled)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/release")
    def release():
        station().release()
        return {"released": True, "message": "Laptop movement cancelled; use handset for pilot takeover"}

    @app.post("/api/incidents")
    def incident(body: Incident):
        return station().incident(body.note)

    @app.get("/api/events")
    def events():
        return station().store.events(100, include_telemetry=False)

    @app.get("/api/export")
    def export():
        return JSONResponse(station().store.events(10000), headers={"Content-Disposition": 'attachment; filename="flight-log.json"'})

    @app.get("/snapshots/{filename}")
    def snapshot(filename: str):
        if not filename.startswith("snapshot-") or not filename.endswith(".jpg") or "/" in filename or "\\" in filename:
            raise HTTPException(404)
        path = station().store.root / filename
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.post("/api/search-grid")
    def grid(body: Grid):
        try:
            points = search_grid(body.latitude, body.longitude, body.width_m, body.height_m, body.spacing_m)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"type": "FeatureCollection", "execution": "Preview/export only; no flight upload",
                "features": [{"type": "Feature", "properties": {"name": "Search route"},
                              "geometry": {"type": "LineString", "coordinates": [[p["longitude"], p["latitude"]] for p in points]}}]}

    app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")
    return app


app = create_app()
