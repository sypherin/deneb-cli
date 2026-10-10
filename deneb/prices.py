"""deneb.prices — live hardware prices, read from the seller's own page every day.

Prices typed into web pages go stale in weeks and then disagree with each other. This module is the one place
a price comes from: deneb/data/price_sources.json names the official stores and
authorised Singapore resellers, and refresh() reads each product page's own
machine-readable price. The Deneb landscape job publishes the result in landscape.json,
which altronis.sg reads, so the site, the Deneb assistant and the CLI quote one number.

Layers:
  * pure:  parse_shopify(), parse_meta(), parse_availability(), parse_apple(), summarise_item()  (unit-tested)
  * I/O:   fetch_source(), fetch_fx(), refresh()                          (network, fail-soft)

Fail-soft per source, loud in the output: a source that cannot be read is kept with
ok=False and its error, never silently dropped, and an item with no readable source
carries no price rather than a guessed one.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES_PATH = os.path.join(_HERE, "data", "price_sources.json")
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128 Safari/537.36"
SGT = _dt.timezone(_dt.timedelta(hours=8))

# Keyless daily FX feeds, tried in order. (url, how to pull USD->SGD out of the JSON)
FX_FEEDS = (
    ("https://api.frankfurter.dev/v1/latest?base=USD&symbols=SGD", lambda d: (d["rates"]["SGD"], d.get("date"))),
    ("https://open.er-api.com/v6/latest/USD", lambda d: (d["rates"]["SGD"], d.get("time_last_update_utc"))),
)


# ── pure ──────────────────────────────────────────────────────────────────────
def _hit(text: str, match, exclude) -> bool:
    t = (text or "").lower()
    if match and not any(m.lower() in t for m in match):
        return False
    return not any(x.lower() in t for x in (exclude or []))


def parse_shopify(product: dict, match=None, exclude=None) -> list:
    """Shopify /products/<handle>.js -> sorted distinct prices (store currency) of the
    variants whose title matches. Shopify gives prices in cents. Unavailable variants
    are skipped: a price you cannot buy at is not a price."""
    out = set()
    for v in (product or {}).get("variants") or []:
        if not v.get("available", True):
            continue
        if not _hit(v.get("title") or "", match, exclude):
            continue
        try:
            out.add(round(float(v["price"]) / 100.0, 2))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out)


_META_AMT = re.compile(r'product:price:amount"\s+content="([\d.,]+)"')
_META_CUR = re.compile(r'product:price:currency"\s+content="([A-Z]{3})"')


def parse_meta(html: str) -> tuple:
    """og product meta tags -> ([price], currency or None)."""
    m = _META_AMT.search(html or "")
    if not m:
        return [], None
    try:
        price = float(m.group(1).replace(",", ""))
    except ValueError:
        return [], None
    c = _META_CUR.search(html or "")
    return [price], (c.group(1) if c else None)


_LD_AVAIL = re.compile(r'"availability"\s*:\s*"(?:https?:\\?/\\?/schema\.org\\?/)?(\w+)"')
_BUYABLE = {"InStock", "PreOrder", "BackOrder", "LimitedAvailability", "OnlineOnly", "InStoreOnly"}


def parse_availability(html: str) -> Optional[bool]:
    """schema.org availability in the page's JSON-LD: True when any offer is buyable
    (in stock, pre-order, back-order), False when every offer is out of stock or sold
    out, None when the page does not say. A shop keeps its og price tag on a sold-out
    product, so the price alone does not mean you can buy at it."""
    vals = set(_LD_AVAIL.findall(html or ""))
    if not vals:
        return None
    return bool(vals & _BUYABLE)


def parse_apple(html: str, match=None) -> list:
    """Apple Store buy page -> sorted prices (SGD on /sg/) of the models whose price key
    matches (keys look like 'm5ultra-36-80'). Apple embeds them as a JSON object
    '"prices":{...}' in the page; we brace-match it rather than regex the numbers."""
    s = html or ""
    i = s.find('"prices":{')
    if i < 0:
        return []
    j = i + len('"prices":')
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                break
    else:
        return []
    try:
        obj = json.loads(s[j:k + 1])
    except ValueError:
        return []
    out = set()
    for key, v in obj.items():
        if match and not any(m.lower() in key.lower() for m in match):
            continue
        try:
            out.add(float(v["amount"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out)


def to_sgd(amount: float, currency: str, usd_sgd: Optional[float]) -> Optional[float]:
    if currency == "SGD":
        return round(amount, 2)
    if currency == "USD" and usd_sgd:
        return round(amount * usd_sgd, 2)
    return None


def summarise_item(item: dict, results: list, gst_rate: float) -> dict:
    """One item's published record. from/to are LANDED SGD: overseas prices get GST at
    import added (still before shipping), local prices already include GST. None when
    no source could be read, never a stale or guessed figure."""
    landed = []
    for r in results:
        if not r.get("ok"):
            continue
        for p in r.get("prices_sgd") or []:
            landed.append(round(p * (1 + gst_rate), 2) if r.get("ships_from") == "overseas" else p)
    return {
        "id": item["id"], "device": item.get("device"), "label": item.get("label"),
        "class": item.get("class"),
        "from_sgd": min(landed) if landed else None,
        "to_sgd": max(landed) if landed else None,
        "sources_ok": sum(1 for r in results if r.get("ok")),
        "sources": results,
    }


# ── I/O ───────────────────────────────────────────────────────────────────────
# Shopify's edge answers 429 on some mornings to every store at once (5 of the 10 daily
# runs from 1 to 10 Oct 2026 lost the GMKtec, Bosgame, SourceIT and Vii PC prices), and
# the same URL reads fine minutes later. So: wait and retry a throttled read, honouring
# Retry-After, and space requests to one host instead of firing them back to back.
_RETRY_STATUS = (429, 503)
_RETRY_WAITS = (10, 30)   # seconds before the 2nd and 3rd attempt, unless the server says
_MAX_WAIT = 60
_HOST_GAP = 2.0           # seconds between two requests to the same host
_RETRY_BUDGET = 180.0     # total retry waiting per run: past it, a throttled read fails
_last_hit: dict = {}
_currency_cache: dict = {}
_budget = {"left": _RETRY_BUDGET}


def retry_after(value: Optional[str], default: float) -> float:
    """Seconds to wait from a Retry-After header (delta-seconds or an HTTP date), kept
    within 1.._MAX_WAIT; `default` when absent or unreadable. Pure, unit-tested."""
    if not value:
        return default
    value = value.strip()
    if value.isdigit():
        secs = float(value)
    else:
        try:
            from email.utils import parsedate_to_datetime
            secs = (parsedate_to_datetime(value) - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            return default
    return min(max(secs, 1.0), float(_MAX_WAIT))


def _fetch(url: str, timeout: int = 25, waits=_RETRY_WAITS, sleep=time.sleep) -> str:
    # store URLs can carry non-ASCII (a "™" in a product handle); percent-encode it,
    # leaving existing escapes and URL punctuation alone
    url = urllib.parse.quote(url, safe=":/?&=%#+,;@!$'()*[]~")
    host = urllib.parse.urlsplit(url).netloc
    for attempt in range(len(waits) + 1):
        gap = _HOST_GAP - (time.monotonic() - _last_hit.get(host, -_HOST_GAP))
        if gap > 0:
            sleep(gap)
        _last_hit[host] = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept-Language": "en-SG,en;q=0.9"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY_STATUS or attempt == len(waits):
                raise
            wait = retry_after(e.headers.get("Retry-After") if e.headers else None, waits[attempt])
            if wait > _budget["left"]:
                raise
            _budget["left"] -= wait
            print(f"[prices] {host} HTTP {e.code}, retry {attempt + 1}/{len(waits)} in {wait:.0f}s",
                  file=sys.stderr, flush=True)
            sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


def _shopify_currency(url: str) -> Optional[str]:
    """The store's currency from /cart.js, read once per store per run (SourceIT lists
    three products; asking three times only adds to the throttling above)."""
    root = re.match(r"(https?://[^/]+)", url).group(1)
    if root not in _currency_cache:
        try:
            _currency_cache[root] = json.loads(_fetch(root + "/cart.js")).get("currency")
        except Exception:  # noqa: BLE001
            return None    # not cached: the next product from this store asks again
    return _currency_cache[root]


def fetch_fx() -> tuple:
    """(usd_sgd, date, source_url) from the first FX feed that answers, else (None, None, None)."""
    for url, pick in FX_FEEDS:
        try:
            rate, date = pick(json.loads(_fetch(url)))
            if rate and 0.8 < float(rate) < 2.0:  # sanity bound: USD/SGD has lived near 1.3
                return float(rate), date, url
        except Exception:  # noqa: BLE001
            continue
    return None, None, None


def fetch_source(src: dict, usd_sgd: Optional[float], now: str) -> dict:
    rec = {k: src.get(k) for k in ("seller", "trust", "url", "ships_from")}
    rec.update({"checked": now, "ok": False})
    kind = src.get("kind")
    if kind == "link":  # the seller blocks automated reads: list the link, never a price
        rec["link_only"] = True
        return rec
    try:
        if kind == "shopify":
            raw = parse_shopify(json.loads(_fetch(src["url"] + ".js")),
                                src.get("variant_match"), src.get("variant_exclude"))
            cur = _shopify_currency(src["url"])
        elif kind == "meta":
            page = _fetch(src["url"])
            raw, cur = parse_meta(page)
            if raw and parse_availability(page) is False:
                rec.update({"out_of_stock": True, "listed": raw[0], "currency": cur})
                raise ValueError(f"out of stock (page still lists {cur or ''} {raw[0]:,.0f})")
        elif kind == "apple":
            raw, cur = parse_apple(_fetch(src["url"]), src.get("variant_match")), "SGD"
        else:
            raise ValueError(f"unknown kind {kind!r}")
        if not raw:
            raise ValueError("no price found on the page")
        if not cur:
            raise ValueError("currency unknown")
        sgd = [to_sgd(p, cur, usd_sgd) for p in raw]
        if any(p is None for p in sgd):
            raise ValueError(f"cannot convert {cur} (FX unavailable)")
        rec.update({"ok": True, "currency": cur, "prices": raw, "prices_sgd": sgd})
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    return rec


def load_sources(path: str = SOURCES_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def refresh(cfg: Optional[dict] = None) -> dict:
    """Read every source now. Returns the publishable block (JSON-safe) with an errors
    list; one dead page never sinks the rest."""
    cfg = cfg or load_sources()
    _currency_cache.clear()
    _budget["left"] = _RETRY_BUDGET
    now = _dt.datetime.now(SGT).isoformat(timespec="minutes")
    usd_sgd, fx_date, fx_src = fetch_fx()
    gst = float(cfg.get("gst_rate", 0.09))
    items, errors, unavailable = [], [], []
    if usd_sgd is None:
        errors.append("fx: no USD/SGD feed answered; overseas prices skipped")
    for it in cfg.get("items") or []:
        results = [fetch_source(s, usd_sgd, now) for s in it.get("sources") or []]
        for r in results:
            if r.get("link_only"):
                continue
            if r.get("out_of_stock"):  # a market state, not a broken feed
                unavailable.append(f"{it['id']} / {r['seller']}: {r.get('error')}")
            elif not r["ok"]:
                errors.append(f"{it['id']} / {r['seller']}: {r.get('error')}")
        items.append(summarise_item(it, results, gst))
    return {"generated": now, "currency": "SGD", "gst_rate": gst,
            "fx": {"usd_sgd": usd_sgd, "date": fx_date, "source": fx_src},
            "items": items, "unavailable": unavailable, "errors": errors}


if __name__ == "__main__":  # pragma: no cover - manual check
    print(json.dumps(refresh(), indent=1))
