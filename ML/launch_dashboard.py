"""Start vision alone, or open the existing local service without loading another model."""
import argparse
import json
import socket
import sys
import webbrowser
from urllib.request import build_opener, ProxyHandler


def health(port):
    try:
        with build_opener(ProxyHandler({})).open(f'http://127.0.0.1:{port}/api/health', timeout=2) as response:
            return json.load(response)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    url = f'http://127.0.0.1:{args.port}/'
    if health(args.port).get('application') == 'AURA':
        print(f'Reusing AURA: {url}. Keep its original service window open.', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return
    # Fail without disturbing another application occupying the requested port.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    from dashboard import serve
    serve(port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Startup failed: {exc}', file=sys.stderr)
        sys.exit(1)
