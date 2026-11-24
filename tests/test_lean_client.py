import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from rci.tasks.fate.lean_service.client import LeanClient


def serve(name: str) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def _send(self, body: dict):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send({"workers": 1})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self._send({"text": name})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def url(srv) -> str:
    return f"http://127.0.0.1:{srv.server_address[1]}"


def test_failover_to_replacement_service(tmp_path):
    a, b = serve("A"), serve("B")
    addr = tmp_path / "current.json"
    addr.write_text(json.dumps({"url": url(a)}))
    c = LeanClient.from_addr_file(str(addr))
    c.failover_poll_s = 0.05
    assert c.search("x")["text"] == "A"
    addr.write_text(json.dumps({"url": url(b)}))
    assert c.search("x")["text"] == "A"
    a.shutdown()
    a.server_close()
    assert c.search("x")["text"] == "B"
    assert c.url == url(b)
    b.shutdown()


def test_no_replacement_raises(tmp_path):
    a = serve("A")
    addr = tmp_path / "current.json"
    addr.write_text(json.dumps({"url": url(a)}))
    c = LeanClient.from_addr_file(str(addr))
    c.failover_wait_s, c.failover_poll_s = 0.3, 0.05
    a.shutdown()
    a.server_close()
    with pytest.raises(requests.ConnectionError):
        c.search("x")


def test_plain_url_client_does_not_fail_over():
    a = serve("A")
    c = LeanClient(url(a))
    a.shutdown()
    a.server_close()
    with pytest.raises(requests.ConnectionError):
        c.search("x")
