"""Serve a built custom frontend alongside a DepthAI application."""

import logging
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
    """Serve a built frontend over HTTP for the duration of the context.

    Wrap device initialization and the pipeline's run loop with this context and
    use ``dai.RemoteConnection(serveFrontend=False)`` for live data. The server
    disables caching, falls back to ``index.html`` for missing extensionless paths,
    and closes on exit, including exceptions.

    Args:
        directory: Directory containing the built ``index.html``. Construct it from
            the application's ``__file__`` to avoid depending on the working directory.
        host: Address to bind. Defaults to loopback for local browser access.
        port: HTTP port, defaulting to 8082. Independent of the WebSocket port.
        enabled: Set false to skip hosting when another server serves the frontend.

    Yields:
        The HTTP URL, or None when frontend hosting is disabled.

    Raises:
        FileNotFoundError: The directory has no built ``index.html``.
        OSError: The server cannot bind to the requested address and port.

    Notes:
        Serves static files only; provides no TLS, authentication, or WebSocket proxy.
    """
    if not enabled:
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
