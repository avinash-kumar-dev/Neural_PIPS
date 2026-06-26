import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime, timezone

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
SIGNALS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'eurusd', 'live_signals.json')


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_GET(self):
        if self.path == '/api/signals':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                if os.path.exists(SIGNALS_FILE):
                    with open(SIGNALS_FILE, 'r') as f:
                        signals = json.load(f)
                else:
                    signals = []

                self.wfile.write(json.dumps({
                    'status': 'ok',
                    'signals': signals,
                    'count': len(signals),
                }).encode())
            except Exception as e:
                self.wfile.write(json.dumps({
                    'status': 'error',
                    'error': str(e),
                    'signals': [],
                }).encode())

        elif self.path == '/api/session':
            now_utc = datetime.now(timezone.utc)
            in_session = 13 <= now_utc.hour < 16

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            self.wfile.write(json.dumps({
                'status': 'ok',
                'in_session': in_session,
                'utc_hour': now_utc.hour,
                'utc_time': now_utc.strftime('%H:%M:%S'),
            }).encode())

        else:
            super().do_GET()

    def log_message(self, format, *args):
        if '/api/' not in str(args[0]):
            super().log_message(format, *args)


def main():
    port = 8080
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{port}")
    print(f"Serving from: {FRONTEND_DIR}")
    print(f"Signals file: {SIGNALS_FILE}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == '__main__':
    main()
