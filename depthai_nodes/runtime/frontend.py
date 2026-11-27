"""Serve a built custom frontend alongside a peripheral DepthAI application."""

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)


class _FrontendHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".css": "text/css",
        ".wasm": "application/wasm",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        # Different examples reuse the same localhost origin.
        del self.headers["If-Modified-Since"]
        path = Path(self.translate_path(self.path))
        if not path.exists() and not path.suffix:
            self.path = "/index.html"
        return super().send_head()

    def log_message(self, format, *args):
        logger.debug(format, *args)


@contextmanager
def serve_frontend(
    directory: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8082,
    enabled: bool = True,
) -> Iterator[str | None]:
    """Serve built frontend assets for the lifetime of an application.

    Use this context around device initialization and the pipeline's run loop, with
    ``dai.RemoteConnection(serveFrontend=False)``. The server closes even if startup
    or processing raises. No server is started when the oakapp runtime sets
    ``OAKAPP_STATIC_FRONTEND_PORT`` or when ``enabled`` is false.

    Args:
        directory: Directory containing the built ``index.html``. Construct it from
            the application's ``__file__`` so it does not depend on the working
            directory. Frontend compilation is an explicit setup step.
        host: Address to bind. Defaults to loopback for local browser access.
        port: HTTP port. Defaults to 8082, matching the viewer's peripheral URL
            convention. This does not change RemoteConnection's WebSocket port.
        enabled: Set false when using a separate frontend development server.

    Yields:
        The HTTP URL, or None when frontend hosting is disabled or managed by oakapp.

    Raises:
        FileNotFoundError: The directory has no built ``index.html``.
        OSError: The server cannot bind to the requested address and port.

    Notes:
        This serves static files over HTTP, without authentication or a WebSocket
        proxy. For remote access to encoded video, use an HTTPS reverse proxy that
        also forwards the DepthAI WebSocket connection. Localhost needs no TLS.
    """
    if not enabled or os.environ.get("OAKAPP_STATIC_FRONTEND_PORT"):
        yield None
        return

    directory = Path(directory).resolve()
    if not (directory / "index.html").is_file():
        raise FileNotFoundError(
            f"Frontend build not found: {directory / 'index.html'}. "
            "Build the frontend first: cd frontend && npm ci && npm run build"
        )

    handler = partial(_FrontendHandler, directory=str(directory))
    with ThreadingHTTPServer((host, port), handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, name="frontend-http", daemon=True
        )
        thread.start()
        display_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
        url = f"http://{display_host}:{server.server_port}"
        logger.info("Frontend available at %s", url)
        try:
            yield url
        finally:
            server.shutdown()
            thread.join()
