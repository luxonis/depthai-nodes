import socket
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from depthai_nodes.runtime import serve_frontend


@pytest.fixture
def frontend(tmp_path, monkeypatch):
    monkeypatch.delenv("OAKAPP_STATIC_FRONTEND_PORT", raising=False)
    (tmp_path / "index.html").write_text("<html>custom frontend</html>")
    (tmp_path / "app.js").write_text("export const ready = true;")
    (tmp_path / "font.woff2").write_bytes(b"font")
    return tmp_path


def test_serves_assets_and_browser_routes(frontend):
    with serve_frontend(frontend, port=0) as url:
        for path in ("/", "/measure/points?unit=mm"):
            with urlopen(url + path, timeout=2) as response:
                assert response.read() == b"<html>custom frontend</html>"
        with urlopen(url + "/app.js", timeout=2) as response:
            assert response.headers.get_content_type() == "application/javascript"
            assert response.read() == b"export const ready = true;"
        with urlopen(url + "/font.woff2", timeout=2) as response:
            assert response.headers.get_content_type() == "font/woff2"
        with pytest.raises(HTTPError) as error:
            urlopen(url + "/missing.js", timeout=2)
        assert error.value.code == 404


def test_missing_build_explains_setup(tmp_path, monkeypatch):
    monkeypatch.delenv("OAKAPP_STATIC_FRONTEND_PORT", raising=False)
    with pytest.raises(FileNotFoundError, match="npm ci && npm run build"):
        with serve_frontend(tmp_path):
            pytest.fail("Unbuilt frontend was accepted")


@pytest.mark.parametrize("standalone", [False, True])
def test_disabled_server_needs_no_bundle_or_free_port(
    tmp_path, monkeypatch, standalone
):
    if standalone:
        monkeypatch.setenv("OAKAPP_STATIC_FRONTEND_PORT", "9000")
    else:
        monkeypatch.delenv("OAKAPP_STATIC_FRONTEND_PORT", raising=False)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with serve_frontend(
            tmp_path, port=occupied.getsockname()[1], enabled=standalone
        ) as url:
            assert url is None


def test_releases_port_after_application_failure(frontend):
    with pytest.raises(RuntimeError, match="pipeline failed"):
        with serve_frontend(frontend, port=0) as url:
            port = int(url.rsplit(":", 1)[1])
            raise RuntimeError("pipeline failed")
    with serve_frontend(frontend, port=port) as url:
        with urlopen(url, timeout=2) as response:
            assert response.status == 200


def test_reused_origin_does_not_reuse_previous_frontend(frontend):
    with serve_frontend(frontend, port=0) as url:
        request = Request(
            url, headers={"If-Modified-Since": "Wed, 01 Jan 2031 00:00:00 GMT"}
        )
        with urlopen(request, timeout=2) as response:
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert response.read() == b"<html>custom frontend</html>"
