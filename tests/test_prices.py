"""deneb.prices — pure parser/summary tests (no network)."""
import json

from deneb import prices as pr


SHOPIFY = {"variants": [
    {"title": "64GB+1TB", "price": 219999, "available": True},
    {"title": "128GB+1TB", "price": 349999, "available": True},
    {"title": "128GB+2TB", "price": 364999, "available": True},
    {"title": "128GB+2TB + Z.AI Coding Plan", "price": 369999, "available": True},
    {"title": "128GB+4TB", "price": 399999, "available": False},
]}


def test_shopify_matches_excludes_and_skips_unavailable():
    got = pr.parse_shopify(SHOPIFY, ["128gb"], ["coding plan"])
    assert got == [3499.99, 3649.99]


def test_shopify_no_match_is_empty_not_everything():
    assert pr.parse_shopify(SHOPIFY, ["256gb"]) == []
    assert pr.parse_shopify({}, ["128gb"]) == []


def test_shopify_bad_price_skipped():
    assert pr.parse_shopify({"variants": [{"title": "128GB", "price": None}]}, ["128gb"]) == []


def test_meta_price_and_currency():
    html = ('<meta property="product:price:amount" content="5,399.00">'
            '<meta property="product:price:currency" content="SGD">')
    assert pr.parse_meta(html) == ([5399.0], "SGD")


def test_meta_missing():
    assert pr.parse_meta("<html></html>") == ([], None)
    assert pr.parse_meta('<meta property="product:price:amount" content="5399">') == ([5399.0], None)


APPLE = ('<script>window.x={"a":1,"prices":{"m5max-18-32":{"amount":3499,"currency":"SGD"},'
         '"m5ultra-30-64":{"amount":7999,"nested":{"k":1}},"m5ultra-36-80":{"amount":9949}},"z":2}</script>')


def test_apple_filters_keys_and_handles_nesting():
    assert pr.parse_apple(APPLE, ["m5ultra"]) == [7999.0, 9949.0]
    assert pr.parse_apple(APPLE) == [3499.0, 7999.0, 9949.0]


def test_apple_absent_or_truncated():
    assert pr.parse_apple("<html>no prices</html>") == []
    assert pr.parse_apple('"prices":{"m5ultra":{"amount":1') == []


def test_to_sgd():
    assert pr.to_sgd(100, "SGD", None) == 100
    assert pr.to_sgd(100, "USD", 1.275) == 127.5
    assert pr.to_sgd(100, "USD", None) is None
    assert pr.to_sgd(100, "EUR", 1.3) is None


def test_summarise_adds_gst_only_for_overseas_and_ignores_failed():
    item = {"id": "x", "device": "d", "label": "L", "class": "small"}
    res = [
        {"ok": True, "ships_from": "overseas", "prices_sgd": [1000.0]},
        {"ok": True, "ships_from": "sg", "prices_sgd": [1500.0]},
        {"ok": False, "ships_from": "sg", "prices_sgd": [1.0], "error": "boom"},
    ]
    s = pr.summarise_item(item, res, 0.09)
    assert s["from_sgd"] == 1090.0 and s["to_sgd"] == 1500.0
    assert s["sources_ok"] == 2 and len(s["sources"]) == 3


def test_summarise_no_source_means_no_price():
    s = pr.summarise_item({"id": "x"}, [{"ok": False, "error": "e"}], 0.09)
    assert s["from_sgd"] is None and s["to_sgd"] is None and s["sources_ok"] == 0


def test_fetch_source_fails_soft(monkeypatch):
    def boom(url, timeout=25):
        raise OSError("down")
    monkeypatch.setattr(pr, "_fetch", boom)
    r = pr.fetch_source({"seller": "S", "kind": "meta", "url": "https://x/p"}, 1.3, "now")
    assert r["ok"] is False and "down" in r["error"]


def test_fetch_source_overseas_without_fx_is_an_error(monkeypatch):
    monkeypatch.setattr(pr, "_fetch", lambda url, timeout=25:
                        json.dumps({"currency": "USD"}) if url.endswith("/cart.js") else json.dumps(SHOPIFY))
    src = {"seller": "G", "kind": "shopify", "url": "https://shop.example/products/p",
           "variant_match": ["128gb"], "variant_exclude": ["coding plan"], "ships_from": "overseas"}
    assert pr.fetch_source(src, None, "now")["ok"] is False
    ok = pr.fetch_source(src, 1.3, "now")
    assert ok["ok"] and ok["currency"] == "USD" and ok["prices_sgd"] == [4549.99, 4744.99]


def test_sources_file_is_valid():
    cfg = pr.load_sources()
    assert 0 < cfg["gst_rate"] < 0.2
    ids = [i["id"] for i in cfg["items"]]
    assert len(ids) == len(set(ids)) and ids
    for it in cfg["items"]:
        assert it["sources"]
        for s in it["sources"]:
            assert s["kind"] in {"shopify", "meta", "apple"}
            assert s["url"].startswith("https://") and s["ships_from"] in {"sg", "overseas"}
