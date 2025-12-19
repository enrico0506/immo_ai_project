from pathlib import Path

from immo_ai.parsing.immobilie1 import parse_item


def test_parse_immobilie1_fixture() -> None:
    html = Path("tests/fixtures/immobilie1_expose.html").read_text(encoding="utf-8")
    item = {"url": "https://www.immobilie1.de/wohnung-12345", "html": html}
    record, reason = parse_item(item, source="immobilie1", run_id="test", bs_parser="lxml")

    assert reason is None
    assert record is not None
    assert record.title == "Helle Wohnung in Citylage"
    assert record.address == "12345 Berlin, Mitte"
    assert record.expose_id == "12345"
    assert record.price_eur == 350000.0
    assert record.area_sqm == 82.5
    assert record.rooms == 3.0


def test_parse_immobilie1_invalid_html_rejects() -> None:
    item = {"url": "https://www.immobilie1.de/wohnung-999", "html": ""}
    record, reason = parse_item(item, source="immobilie1", run_id="test", bs_parser="lxml")

    assert record is None
    assert reason == "missing_html"
