"""Fake servers for the tests: an OpenAI-compatible model server and a Deneb engine, both on
127.0.0.1 in the port range 18740-18760 (an ephemeral port if all are busy). Nothing here touches the network beyond loopback."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PORT_RANGE = range(18740, 18761)


def _serve(host, handler):
    """Bind inside the reserved test range 18740-18760 (first free), else an ephemeral port."""
    for port in PORT_RANGE:
        try:
            return ThreadingHTTPServer((host, port), handler)
        except OSError:
            continue
    return ThreadingHTTPServer((host, 0), handler)


class _Quiet(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D401
        pass

    def _send(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))


class FakeOpenAI:
    """replies: list of assistant message strings returned in order (the last repeats)."""

    def __init__(self, replies, models=("qwen3.8-27b",), vision=False, host="127.0.0.1"):
        self.replies = list(replies)
        self.models = list(models)
        self.vision = vision
        self.requests = []      # (path, headers dict, json body)
        outer = self

        class H(_Quiet):
            def do_GET(self):
                outer.requests.append((self.path, dict(self.headers), None))
                if self.path.rstrip("/") == "/v1/models":
                    self._send(200, {"object": "list", "data": [
                        {"id": m, "object": "model"} for m in outer.models]})
                else:
                    self._send(404, {"error": "no"})

            def do_POST(self):
                body = json.loads(self._body() or b"{}")
                outer.requests.append((self.path, dict(self.headers), body))
                if self.path.rstrip("/") != "/v1/chat/completions":
                    return self._send(404, {"error": "no"})
                if not outer.vision and "image_url" in json.dumps(body):
                    return self._send(400, {"error": "images not supported"})
                i = min(len([r for r in outer.requests if r[2] is not None]) - 1,
                        len(outer.replies) - 1)
                self._send(200, {"choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant",
                                                          "content": outer.replies[i]}}]})

        self.httpd = _serve(host, H)
        self.port = self.httpd.server_address[1]
        self.base = f"http://{host}:{self.port}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def chats(self):
        return [r for r in self.requests if r[2] is not None]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeEngine:
    """POST /token-request, GET /token-request/<id> as the engine spec writes them."""

    def __init__(self):
        self.posts = []         # parsed bodies
        self.post_headers = []
        self.status = "pending"
        self.token = "dnb_faketoken"
        self.post_code = 202
        self.post_body = None   # override the response body
        self.poll_code = None   # force a status code on poll
        self.poll_headers = []
        self.claimed = False
        self.authed_code = None     # force a status on the authed routes (/agent-step ...)
        self.authed_body = None
        outer = self

        class H(_Quiet):
            def do_POST(self):
                raw = self._body()
                if self.path in ("/agent-step", "/ask", "/summarize", "/escalate"):
                    return self._send(outer.authed_code or 200, outer.authed_body or {})
                if self.path != "/token-request":
                    return self._send(404, {"error": "no"})
                outer.post_headers.append(dict(self.headers))
                if outer.post_code != 202:
                    return self._send(outer.post_code, outer.post_body or {"error": "nope"})
                outer.posts.append(json.loads(raw))
                self._send(202, outer.post_body or {"id": "abc123XYZ9", "status": "pending",
                                                     "poll_after_s": 30})

            def do_GET(self):
                outer.poll_headers.append(dict(self.headers))
                if not self.path.startswith("/token-request/"):
                    return self._send(404, {"error": "no"})
                if outer.poll_code:
                    return self._send(outer.poll_code, {"status": "unknown"} if outer.poll_code == 404
                                      else {"error": "too many requests, try later"})
                if outer.status == "approved":
                    if outer.claimed:
                        return self._send(200, {"status": "claimed"})
                    outer.claimed = True
                    return self._send(200, {"status": "approved", "token": outer.token})
                self._send(200, {"status": outer.status})

        self.httpd = _serve("127.0.0.1", H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeRedirector:
    """Answers every request with a redirect to `target` (a full URL), or to `target` + path
    when `keep_path` is set. Records the headers it saw."""

    def __init__(self, target=None, code=302, keep_path=False, same_origin=False):
        """same_origin: redirect to /final<path> on this very server, and answer that with a
        small JSON 200 (a model list that is also a valid token-request poll)."""
        self.target, self.code, self.keep_path, self.same_origin = target, code, keep_path, same_origin
        self.seen = []
        outer = self

        class H(_Quiet):
            def _go(self):
                outer.seen.append((self.command, self.path, dict(self.headers)))
                if self.command == "POST":
                    self._body()
                if outer.same_origin and self.path.startswith("/final"):
                    return self._send(200, {"data": [{"id": "m1"}], "status": "pending"})
                if outer.same_origin:
                    loc = f"http://127.0.0.1:{outer.port}/final{self.path}"
                    self.send_response(outer.code)
                    self.send_header("Location", loc)
                    self.send_header("Content-Length", "0")
                    return self.end_headers()
                loc = outer.target + (self.path if outer.keep_path else "")
                self.send_response(outer.code)
                self.send_header("Location", loc)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST = _go

        self.httpd = _serve("127.0.0.1", H)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeRecorder:
    """Records every request (method, path, headers) and answers 200 with `reply`."""

    def __init__(self, reply=None, code=200):
        self.code = code
        self.reply = reply if reply is not None else {"ok": True, "data": []}
        self.seen = []
        outer = self

        class H(_Quiet):
            def _go(self):
                self._body()
                outer.seen.append((self.command, self.path, dict(self.headers)))
                self._send(outer.code, outer.reply)

            do_GET = do_POST = _go

        self.httpd = _serve("127.0.0.1", H)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeBlackHole:
    """Accepts connections at the OS level and never answers (a firewalled engine)."""

    def __init__(self):
        import socket
        self.sock = socket.socket()
        for port in PORT_RANGE:
            try:
                self.sock.bind(("127.0.0.1", port))
                break
            except OSError:
                continue
        else:
            self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.sock.close()
