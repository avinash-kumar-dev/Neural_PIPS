import os
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
SIGNALS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'eurusd', 'live_signals.json')

_fetcher = None


def get_fetcher():
    global _fetcher
    if _fetcher is None:
        from src.data_pipeline import DataFetcher
        _fetcher = DataFetcher(host='localhost', port=8001)
        try:
            _fetcher.connect()
        except Exception:
            _fetcher = None
    return _fetcher


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

        elif self.path.startswith('/api/ohlcv/latest'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                params = {}
                if '?' in self.path:
                    params = dict(p.split('=') for p in self.path.split('?')[1].split('&'))

                timeframe = params.get('tf', 'M5')

                fetcher = get_fetcher()
                if fetcher is None:
                    self.wfile.write(json.dumps({'status': 'error', 'error': 'MT5 not connected'}).encode())
                    return

                df = fetcher.fetch_ohlcv(timeframe, 1)

                if df is None or len(df) == 0:
                    self.wfile.write(json.dumps({'status': 'error', 'error': 'No data'}).encode())
                    return

                row = df.iloc[-1]
                candle = {
                    'time': int(row['time'].timestamp()) if hasattr(row['time'], 'timestamp') else int(datetime.fromisoformat(str(row['time'])).timestamp()),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['tick_volume']),
                }

                self.wfile.write(json.dumps({
                    'status': 'ok',
                    'timeframe': timeframe,
                    'candle': candle,
                }).encode())

            except Exception as e:
                self.wfile.write(json.dumps({'status': 'error', 'error': str(e)}).encode())

        elif self.path.startswith('/api/ohlcv') and not self.path.startswith('/api/ohlcv/latest'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                params = {}
                if '?' in self.path:
                    params = dict(p.split('=') for p in self.path.split('?')[1].split('&'))

                timeframe = params.get('tf', 'M5')
                count = min(int(params.get('count', '200')), 1000)

                fetcher = get_fetcher()
                if fetcher is None:
                    self.wfile.write(json.dumps({'status': 'error', 'error': 'MT5 not connected'}).encode())
                    return

                from src.data_pipeline import TIMEFRAME_MAP
                mt5_tf = TIMEFRAME_MAP.get(timeframe, 5)
                df = fetcher.fetch_ohlcv(timeframe, count)

                if df is None or len(df) == 0:
                    self.wfile.write(json.dumps({'status': 'error', 'error': 'No data'}).encode())
                    return

                candles = []
                for _, row in df.iterrows():
                    candles.append({
                        'time': int(row['time'].timestamp()) if hasattr(row['time'], 'timestamp') else int(datetime.fromisoformat(str(row['time'])).timestamp()),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['tick_volume']),
                    })

                self.wfile.write(json.dumps({
                    'status': 'ok',
                    'timeframe': timeframe,
                    'count': len(candles),
                    'candles': candles,
                }).encode())

            except Exception as e:
                self.wfile.write(json.dumps({'status': 'error', 'error': str(e)}).encode())

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
