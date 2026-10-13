"""deneb.net — the one HTTP helper. client.py, brain.py and tokenreq.py all go through it, so
there is one User-Agent, one timeout policy, one redirect rule and one way to describe a network
failure.

Redirect rule: a redirect that changes scheme, host or port is REFUSED, because the request
carries a key or a secret. Same-origin redirects are followed.
"""
from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.parse
import urllib.request

from . import __version__

# Some gateways block the default "Python-urllib" User-Agent as a bot signature.
UA = f"deneb-cli/{__version__}"

TIMEOUT = 20        # default seconds for a plain request
POLL_TIMEOUT = 3    # a background poll during a chat must never hold the chat up

_PORTS = {"http": 80, "https": 443}


class NetError(Exception):
    """No usable answer from the network (refused, timed out, DNS, bad redirect). The text is
    the reason only."""


class RedirectRefused(NetError):
    pass


def display_host(url: str) -> str:
    """host[:port] of a URL for showing to a person. Userinfo (user:pass@) is never shown."""
    p = urllib.parse.urlparse(url or "")
    try:
        host, port = p.hostname, p.port
    except ValueError:
        return url
    if not host:
        return url
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{port}" if port else host


def _origin(url: str):
    p = urllib.parse.urlparse(url)
    scheme = (p.scheme or "").lower()
    try:
        port = p.port or _PORTS.get(scheme)
    except ValueError:
        port = None
    return scheme, (p.hostname or "").lower(), port


class _SameOriginOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _origin(req.full_url) != _origin(newurl):
            raise RedirectRefused(
                f"the server redirected to another address ({display_host(newurl)}). deneb "
                "does not send your key there. use the final URL instead.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SameOriginOnly)
# for this box's own ports: no proxy from the environment, the way check's curl runs
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SameOriginOnly)


def local_get(port: int, path: str, timeout: float = 3, cap: int = 8000):
    """GET http://localhost:<port><path> with no proxy: check's probe on a box without curl.
    The body as text, an HTTP error's body too (curl -s prints that), capped at `cap`
    characters; None when nothing answered."""
    req = urllib.request.Request(f"http://localhost:{int(port)}{path}",
                                 headers={"User-Agent": UA})
    try:
        with _LOCAL_OPENER.open(req, timeout=timeout) as r:
            raw = r.read(cap * 4)
    except urllib.error.HTTPError as e:
        try:
            raw = e.read(cap * 4)
        except Exception:  # noqa: BLE001 - an error page we can't read is an empty one
            raw = b""
    except (NetError, urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
        return None
    return raw.decode("utf-8", "replace")[:cap]


def request(method: str, url: str, body=None, headers=None, timeout: float = TIMEOUT):
    """(status, raw bytes). An HTTP error status is RETURNED, not raised. Raises NetError when
    there is no answer."""
    h = {"User-Agent": UA}
    h.update(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return r.status, r.read()
    except NetError:
        raise
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
        except Exception:  # noqa: BLE001
            raw = b""
        return e.code, raw
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
        raise NetError(str(getattr(e, "reason", e))) from e


def json_obj(raw: bytes) -> dict:
    """The body as a dict, or {} when it is empty, not JSON, or not an object."""
    try:
        obj = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


def scrub(text, secrets=(), cap: int = 200) -> str:
    """Server text that is safe to show: any secret value and any `Bearer ...` token removed,
    capped at `cap` characters."""
    s = str(text or "")
    for v in secrets:
        if v:
            s = s.replace(v, "[removed]")
    return _BEARER.sub("Bearer [removed]", s)[:cap]
