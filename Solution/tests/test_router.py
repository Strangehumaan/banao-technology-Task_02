"""Smoke tests: text handling, the endpoint contract, and the polite failure.

    uv run pytest          (from the Solution folder)
"""
import pytest
from fastapi.testclient import TestClient

from kestrel_router import data as D
from kestrel_router.features import is_vague, last_clause


def test_clean_text_repairs_legacy_encoding():
    raw = "urgÃ©nt: claim status paid by emi Ã¢â‚¬Â¦ order KO2604629"
    assert D.clean_text(raw) == "urgent: claim status paid by emi ... order idnum"


def test_last_clause_and_vague():
    assert last_clause("hi, is fan covered under shield plan, fan making loud noise") == "fan making loud noise"
    assert is_vague("please call back regarding water purifier asap")
    assert not is_vague("water purifier leaking water from bottom")


@pytest.fixture(scope="module")
def client():
    from service.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    h = client.get("/health").json()
    assert h["model_loaded"] and h["paid_api_calls"] == 0


def test_route_clear_request(client):
    r = client.post("/route", json={"request_text": "air fryer showing error code E7",
                                    "product_family": "Air Fryer"})
    assert r.status_code == 200
    body = r.json()
    assert body["team"] == "Repairs" and body["action"] == "route"
    assert body["reasons"] and body["team"] in D.TEAMS


def test_paid_is_not_billing(client):
    body = client.post("/route", json={
        "request_text": "installer did not turn up for room heater paid on upi",
        "product_family": "Room Heater"}).json()
    assert body["team"] == "Installs & Demo"
    assert any("not Billing" in r for r in body["reasons"])


def test_comma_before_filler_is_not_a_second_request(client):
    body = client.post("/route", json={
        "request_text": "installer did not turn up for room heater, paid on upi",
        "product_family": "Room Heater"}).json()
    assert body["team"] == "Installs & Demo" and body["action"] == "route"


def test_vague_request_asks_first(client):
    body = client.post("/route", json={"request_text": "please call back regarding cooktop asap",
                                       "product_family": "Induction Cooktop"}).json()
    assert body["action"] == "clarify_first" and body["clarifying_question"]


def test_bad_input_is_rejected(client):
    assert client.post("/route", json={"request_text": "x"}).status_code == 422
    assert client.post("/route", json={"request_text": "fan noise", "channel": "fax"}).status_code == 422


def test_polite_failure_without_model(client):
    from service import app as svc
    saved = dict(svc.STATE)
    svc.STATE.update(pipeline=None, error="model file missing")
    try:
        r = client.post("/route", json={"request_text": "fan making loud noise"})
        assert r.status_code == 503
        assert "not routed" in r.json()["error"] and r.json()["what_to_do"]
        assert client.get("/health").json()["status"] == "degraded"
    finally:
        svc.STATE.update(saved)
