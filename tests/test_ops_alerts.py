"""
Alert delivery (2026-09-10): Prometheus rules -> Alertmanager -> owner push
(+ email for critical). Before this, monitoring/alerts.yml fired into
nothing -- there was no Alertmanager.

What must hold:

  * FAIL CLOSED -- no token configured means 503, a wrong token 401. The
    route is reachable through the public proxy.
  * CRITICAL WAKES THE OWNER -- critical firing -> urgent (bypasses mute and
    quiet hours); warnings -> default; resolved -> low.
  * EVERY RULE ROUTES -- each rule carries a severity Alertmanager matches
    on and a summary the push shows.
  * NO SECRETS IN GIT -- the Alertmanager config is a template; addresses
    and passwords arrive from the .env at container start.
"""

import os
import re

import pytest
import yaml

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi.testclient import TestClient

from backend.main import app
from backend.routes import ops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _payload(status="firing", severity="critical", n=1, name="CanslimContainerGone"):
    alerts = [{"status": status, "labels": {"alertname": name, "severity": severity},
               "annotations": {"summary": f"summary {i}", "description": f"desc {i}"}}
              for i in range(n)]
    return {"version": "4", "status": status, "receiver": "app-push",
            "commonLabels": {"alertname": name, "severity": severity}, "alerts": alerts}


@pytest.fixture
def sent(monkeypatch):
    calls = []
    import backend.email_utils as eu
    monkeypatch.setattr(eu, "send_ops_alert", lambda *a, **k: calls.append((a, k)) or True)
    return calls


class TestWebhook:

    def test_no_token_configured_fails_closed(self, monkeypatch, sent):
        monkeypatch.delenv("OPS_WEBHOOK_TOKEN", raising=False)
        r = TestClient(app).post("/api/ops/alerts", json=_payload(),
                                 headers={"Authorization": "Bearer anything"})
        assert r.status_code == 503 and sent == []

    @pytest.mark.parametrize("header", [None, "Bearer wrong", "tok", "Bearer tok "])
    def test_wrong_or_missing_token_is_401(self, monkeypatch, sent, header):
        monkeypatch.setenv("OPS_WEBHOOK_TOKEN", "tok")
        headers = {"Authorization": header} if header else {}
        r = TestClient(app).post("/api/ops/alerts", json=_payload(), headers=headers)
        assert r.status_code == 401 and sent == []

    def test_critical_firing_is_an_urgent_push(self, monkeypatch, sent):
        monkeypatch.setenv("OPS_WEBHOOK_TOKEN", "tok")
        r = TestClient(app).post("/api/ops/alerts", json=_payload(),
                                 headers={"Authorization": "Bearer tok"})
        assert r.status_code == 200, r.text
        (title, message), kw = sent[0]
        assert title == "CRITICAL: CanslimContainerGone"
        assert kw["priority"] == "urgent" and "summary 0 -- desc 0" in message


class TestFormat:

    def test_warning_waits_for_morning(self):
        note = ops.format_alert_group(_payload(severity="warning", name="CanslimMemoryWillHitCap"))
        assert note["title"] == "Warning: CanslimMemoryWillHitCap" and note["priority"] == "default"

    def test_resolved_is_low_and_says_so(self):
        note = ops.format_alert_group(_payload(status="resolved"))
        assert note["title"] == "Resolved: CanslimContainerGone" and note["priority"] == "low"
        assert "desc" not in note["message"]          # no stale "why" on an all-clear

    def test_long_groups_are_capped(self):
        note = ops.format_alert_group(_payload(n=8))
        assert note["message"].count("summary") == ops.MAX_ALERTS_LISTED
        assert "(+3 more)" in note["message"]


class TestMonitoringConfig:

    def _rules(self):
        doc = yaml.safe_load(open(os.path.join(ROOT, "monitoring/alerts.yml")))
        return [r for g in doc["groups"] for r in g["rules"]]

    def test_every_rule_routes_and_explains_itself(self):
        rules = self._rules()
        assert len(rules) >= 7
        for r in rules:
            assert r["labels"]["severity"] in ("warning", "critical"), r["alert"]
            assert r["annotations"].get("summary"), r["alert"]

    def test_memory_trend_rules_ignore_the_post_restart_warm_up(self):
        # Warm-up climbs ~130 MB/h for hours; a trend rule without the
        # uptime gate pages after every deploy (caught live on the first
        # delivered evaluation, 2026-09-10).
        rules = {r["alert"]: r["expr"] for r in self._rules()}
        for name in ("CanslimMemoryClimbing", "CanslimMemoryWillHitCap"):
            assert "container_start_time_seconds" in rules[name], name
            assert "8 * 3600" in rules[name], name

    def test_the_alert_that_cannot_push_goes_by_email(self):
        am = yaml.safe_load(open(os.path.join(ROOT, "monitoring/alertmanager.yml")))
        crit = {r["alert"] for r in self._rules() if r["labels"]["severity"] == "critical"}
        assert "CanslimContainerGone" in crit
        [route] = [r for r in am["route"]["routes"] if 'severity="critical"' in r["matchers"]]
        [recv] = [x for x in am["receivers"] if x["name"] == route["receiver"]]
        assert recv.get("email_configs") and recv.get("webhook_configs")

    def test_prometheus_delivers_to_alertmanager(self):
        prom = yaml.safe_load(open(os.path.join(ROOT, "monitoring/prometheus.yml")))
        targets = prom["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"]
        assert targets == ["alertmanager:9093"]

    def test_no_secrets_or_addresses_in_git(self):
        text = open(os.path.join(ROOT, "monitoring/alertmanager.yml")).read()
        assert "__SMTP_USER__" in text and "__SMTP_TO__" in text
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)   # no email address
        assert "password_file" in text and "credentials_file" in text

    def test_entrypoint_is_one_line(self):
        # docker-compose.yml is CRLF: a multi-line shell block risks a \r
        # landing inside the secret files the entrypoint writes.
        svc = yaml.safe_load(open(os.path.join(ROOT, "docker-compose.yml")))["services"]["alertmanager"]
        [cmd] = svc["command"]
        assert "\n" not in cmd and "\r" not in cmd
        assert "$$AM_SMTP_PASSWORD" in cmd and "$$AM_WEBHOOK_TOKEN" in cmd
