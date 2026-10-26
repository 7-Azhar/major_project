"""Same-origin, token-protected UI bridge to a fixed loopback AURA service."""
from pathlib import Path
import html
import httpx
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, RedirectResponse

VISION = Path(__file__).resolve().parents[2] / 'ML' / 'dashboard'


def install_vision_routes(app, token):
    @app.get('/vision')
    def redirect():
        return RedirectResponse('/vision/')

    @app.get('/vision/')
    def index():
        page = (VISION / 'index.html').read_text(encoding='utf-8')
        return HTMLResponse(page.replace('<body>', '<body data-station-token="' + html.escape(token, quote=True) + '">'))

    @app.get('/vision/{asset}')
    def asset(asset: str):
        if asset not in ('app.js', 'style.css'):
            raise HTTPException(404)
        return FileResponse(VISION / asset)

    @app.api_route('/vision/api/{action}', methods=['GET', 'POST'])
    async def proxy(action: str, request: Request):
        permitted = {'GET': {'frame', 'options', 'health'}, 'POST': {'control', 'upload'}}
        if action not in permitted[request.method]:
            raise HTTPException(404)
        if request.app.state.station.config.video_backend != 'aura':
            raise HTTPException(409, 'Start the AURA integration configuration to use vision controls')
        headers = {}
        content = None
        if request.method == 'POST':
            try:
                size = int(request.headers.get('content-length', '0'))
            except ValueError:
                raise HTTPException(400, 'Invalid request size')
            limit = 1024**3 if action == 'upload' else 4096
            if not 0 < size <= limit:
                raise HTTPException(413, 'Invalid request size')
            headers = {k: request.headers[k] for k in ('content-type', 'content-length', 'x-filename') if k in request.headers}
            # Bounded streaming: no whole-video buffering in the ground station.
            content = request.stream()
        url = f'http://127.0.0.1:{request.app.state.station.config.aura_port}/api/{action}'
        try:
            async with httpx.AsyncClient(timeout=70 if action == 'upload' else 6, trust_env=False) as client:
                result = await client.request(request.method, url, params=request.query_params,
                                              headers=headers, content=content)
            return Response(result.content, status_code=result.status_code,
                            media_type=result.headers.get('content-type', 'application/json'))
        except httpx.HTTPError:
            return Response('{"error":"AURA is unavailable. Start AURA or use Start Ground Station.cmd."}',
                            status_code=503, media_type='application/json')
