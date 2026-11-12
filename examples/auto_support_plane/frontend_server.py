"""Serve the app's buildless frontend (adapted from oak-examples/raw-stream)."""

import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def start_frontend(
    directory: Path, address: str, port: int, websocket_port: int, simulation=False
):
    if not (directory / "index.html").is_file():
        raise FileNotFoundError(f"Frontend index.html is missing from {directory}")

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/connection.json":
                content = json.dumps(
                    {"websocket_port": websocket_port, "simulation": simulation}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
            else:
                super().do_GET()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(
        (address, port), partial(Handler, directory=str(directory))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
