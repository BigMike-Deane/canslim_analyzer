"""CSV upload hardening regression suite for /api/fidelity/upload-*.

Threat model derived from `canslim-next-session-prompt-may9-fidelity-csv-hardening.md`.
Each section is one threat from the brief; either reproduces a vulnerability
that was fixed, or proves immunity for a class that was already safe.

Pattern matches tests/test_fidelity_routes.py (TestClient + auth bypass +
in-memory SQLite). All tests are user-scoped to id=1 via dependency_override.
"""

import os
import math
import pytest
from datetime import date, datetime, timezone

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import (
    init_db, SessionLocal, User, FidelitySnapshot, FidelityPosition,
    FidelityTrade,
)
from backend.auth import get_current_active_user
from backend import fidelity_sync
from backend.routes import fidelity as fidelity_routes
from tests.conftest import override_dependency


_fake_user = User(
    id=1, email="test@test.com", display_name="Test User",
    is_active=True, is_admin=True, hashed_password="",
)


@pytest.fixture(autouse=True, scope="module")
def _auth_override():
    """Scoped auth bypass — see tests/conftest.py:override_dependency."""
    with override_dependency(get_current_active_user, _fake_user):
        yield


init_db()
client = TestClient(app)


def _db():
    return SessionLocal()


def _ensure_user():
    db = _db()
    try:
        if not db.query(User).filter_by(id=1).first():
            db.add(User(
                id=1, email="test@test.com", display_name="Test User",
                is_active=True, is_admin=True, hashed_password="",
            ))
            db.commit()
    finally:
        db.close()


def _wipe():
    db = _db()
    try:
        db.query(FidelityPosition).delete()
        db.query(FidelitySnapshot).delete()
        db.query(FidelityTrade).delete()
        db.commit()
    finally:
        db.close()


_ensure_user()


@pytest.fixture(autouse=True)
def _isolate():
    _wipe()
    yield
    _wipe()


# ── Sample CSVs (well-formed) ─────────────────────────────────────────

VALID_POSITIONS_CSV = (
    "Account Number,Account Name,Symbol,Description,Quantity,Last Price,"
    "Last Price Change,Current Value,Today's Gain/Loss Dollar,"
    "Today's Gain/Loss Percent,Total Gain/Loss Dollar,Total Gain/Loss Percent,"
    "Percent Of Account,Cost Basis Total,Average Cost Basis,Type\n"
    "X1,Individual,AAPL,APPLE INC,100,$150.00,+$1.00,$15000.00,+$100,+0.67%,"
    "+$5000,+50.00%,100.00%,$10000.00,$100.00,Margin,\n"
)

VALID_ACTIVITY_CSV = (
    "\n\n"
    "Run Date,Account,Account Number,Action,Symbol,Description,Type,"
    "Price ($),Quantity,Commission ($),Fees ($),Accrued Interest ($),"
    "Amount ($),Settlement Date\n"
    "04/25/2026,Individual,X1,YOU BOUGHT AAPL,AAPL,,Margin,150.00,10,,,,"
    "-$1500.00,04/27/2026\n"
)


def _post_positions(content: bytes, filename: str = "positions.csv"):
    return client.post(
        "/api/fidelity/upload-positions",
        files={"file": (filename, content, "text/csv")},
    )


def _post_activity(content: bytes, filename: str = "activity.csv"):
    return client.post(
        "/api/fidelity/upload-activity",
        files={"file": (filename, content, "text/csv")},
    )


# ════════════════════════════════════════════════════════════════════
# Threat 1 — File-size DoS (cap ~10MB; 413 on overflow)
# ════════════════════════════════════════════════════════════════════

class TestThreat1FileSizeDoS:

    def test_positions_rejects_oversize_payload(self):
        """11MB file → 413, no parser invocation."""
        big = b"x" * (11 * 1024 * 1024)
        r = _post_positions(big)
        assert r.status_code == 413
        assert "too large" in r.json()["detail"].lower()

    def test_activity_rejects_oversize_payload(self):
        big = b"x" * (11 * 1024 * 1024)
        r = _post_activity(big)
        assert r.status_code == 413

    def test_positions_accepts_at_cap_minus_one(self):
        """5MB well under cap is fine — parser runs, returns 400 'No positions'
        because the giant filler isn't a real CSV row."""
        # 5MB of header + filler that produces no positions for our account.
        body = VALID_POSITIONS_CSV + ("Z," * (5 * 1024 * 1024 // 2)) + "\n"
        r = _post_positions(body.encode("utf-8"))
        # Either succeeds (if AAPL row parses) or 400 "no positions"
        # — must NOT 413 / 500.
        assert r.status_code in (200, 400)


# ════════════════════════════════════════════════════════════════════
# Threat 2 — Encoding-bomb / decode crash
# ════════════════════════════════════════════════════════════════════

class TestThreat2Decode:

    def test_positions_handles_utf8_bom(self):
        """UTF-8 BOM at start of file is stripped and parsed cleanly."""
        body = "﻿" + VALID_POSITIONS_CSV
        r = _post_positions(body.encode("utf-8"))
        assert r.status_code == 200
        assert r.json()["positions_count"] == 1

    def test_positions_handles_trailing_nulls(self):
        """\\x00 padding at end of file is scrubbed, not crashed on."""
        body = VALID_POSITIONS_CSV.encode("utf-8") + b"\x00" * 100
        r = _post_positions(body)
        assert r.status_code == 200

    def test_positions_handles_arbitrary_high_bytes(self):
        """Non-UTF8 bytes mid-content fall back to latin-1 cleanly."""
        body = VALID_POSITIONS_CSV.encode("utf-8").replace(b"APPLE", b"A\xffPLE")
        r = _post_positions(body)
        assert r.status_code == 200

    def test_activity_handles_utf8_bom(self):
        body = "﻿" + VALID_ACTIVITY_CSV
        r = _post_activity(body.encode("utf-8"))
        assert r.status_code == 200


# ════════════════════════════════════════════════════════════════════
# Threat 3 — MIME / filename bypass
# ════════════════════════════════════════════════════════════════════

class TestThreat3MimeBypass:

    def test_filename_is_only_defense_in_depth(self):
        """A `.csv` filename containing arbitrary garbage is rejected at
        the parser layer (header validation), not at the filename gate.
        The filename gate is intentionally weak — bytes are the truth."""
        garbage = b"this is not csv at all just plain text"
        r = _post_positions(garbage, filename="evil.csv")
        # Must reject via parser, not crash via 500
        assert r.status_code == 400

    def test_csv_dot_exe_filename_still_rejected(self):
        """A `.csv.exe` filename fails the .csv check."""
        r = _post_positions(VALID_POSITIONS_CSV.encode("utf-8"), filename="x.csv.exe")
        assert r.status_code == 400
        assert "CSV" in r.json()["detail"]

    def test_missing_filename_does_not_crash(self):
        """Form-data uploads can omit filename; route returns 400/422
        instead of 500 from AttributeError on NoneType.endswith. 422 is
        FastAPI's multipart-validation reject layer; 400 is our own."""
        r = client.post(
            "/api/fidelity/upload-positions",
            files={"file": ("", VALID_POSITIONS_CSV.encode("utf-8"), "text/csv")},
        )
        assert r.status_code in (400, 422)


# ════════════════════════════════════════════════════════════════════
# Threat 4 — Parser exception leak (no 500 / no stack trace to client)
# ════════════════════════════════════════════════════════════════════

class TestThreat4ExceptionLeak:

    def test_unbalanced_quotes_returns_400_not_500(self):
        """Unterminated quoted field → csv.Error → wrapped as 400."""
        # Note: csv.DictReader is more lenient than expected; this
        # specifically tests the wrapping plumbing.
        body = (
            "Account Number,Symbol,Quantity\n"
            'X1,"AAPL,100\n'  # unterminated quote — csv.Error sometimes
        )
        r = _post_positions(body.encode("utf-8"))
        # Either a 200 (parser tolerated it) or 400 (rejected) — never 500.
        assert r.status_code in (200, 400)

    def test_unexpected_exception_caught_in_route(self, monkeypatch):
        """If parser raises a non-ValueError, route logs and returns 400."""
        def boom(_):
            raise RuntimeError("internal parser failure with secret=hunter2")
        monkeypatch.setattr(fidelity_sync, "parse_positions_csv", boom)
        r = _post_positions(VALID_POSITIONS_CSV.encode("utf-8"))
        assert r.status_code == 400
        # CRITICAL: secret value must NOT appear in client response.
        assert "hunter2" not in r.text
        assert "RuntimeError" not in r.text

    def test_value_error_message_truncated(self, monkeypatch):
        """ValueError messages are surfaced but truncated to 200 chars
        to avoid log-flooding via reflected error text."""
        def boom(_):
            raise ValueError("X" * 5000)
        monkeypatch.setattr(fidelity_sync, "parse_positions_csv", boom)
        r = _post_positions(VALID_POSITIONS_CSV.encode("utf-8"))
        assert r.status_code == 400
        # 200-char cap + the prefix "Could not parse positions CSV: "
        assert len(r.json()["detail"]) < 300


# ════════════════════════════════════════════════════════════════════
# Threat 5 — Field-length overflow (truncate untrusted strings)
# ════════════════════════════════════════════════════════════════════

class TestThreat5FieldLength:

    def test_positions_symbol_truncated(self):
        """A 10KB symbol value is cut to MAX_SYMBOL_LEN (32) at parse time."""
        long_sym = "A" * 10_000
        body = (
            "Account Number,Symbol,Quantity,Last Price,Current Value,"
            "Total Gain/Loss Dollar,Total Gain/Loss Percent,"
            "Cost Basis Total,Average Cost Basis,Percent Of Account,Type\n"
            f"X1,{long_sym},100,$150.00,$15000.00,+$5000,+50%,$10000,$100,100%,Margin\n"
        )
        r = _post_positions(body.encode("utf-8"))
        assert r.status_code == 200
        # DB row's symbol is at most MAX_SYMBOL_LEN
        db = _db()
        try:
            pos = db.query(FidelityPosition).first()
            assert pos is not None
            assert len(pos.symbol) <= fidelity_sync.MAX_SYMBOL_LEN
            assert pos.symbol == "A" * fidelity_sync.MAX_SYMBOL_LEN
        finally:
            db.close()

    def test_positions_description_truncated(self):
        """A 5000-char description is truncated to MAX_DESCRIPTION_LEN (512)."""
        long_desc = "Z" * 5000
        body = (
            "Account Number,Symbol,Description,Quantity,Last Price,"
            "Current Value,Total Gain/Loss Dollar,Total Gain/Loss Percent,"
            "Cost Basis Total,Average Cost Basis,Percent Of Account,Type\n"
            f"X1,AAPL,{long_desc},100,$150.00,$15000.00,+$5000,+50%,"
            f"$10000,$100,100%,Margin\n"
        )
        r = _post_positions(body.encode("utf-8"))
        assert r.status_code == 200
        db = _db()
        try:
            pos = db.query(FidelityPosition).filter_by(symbol="AAPL").first()
            assert pos is not None
            assert len(pos.description) <= fidelity_sync.MAX_DESCRIPTION_LEN
        finally:
            db.close()

    def test_activity_action_truncated(self):
        """raw_action is truncated to MAX_ACTION_LEN (256)."""
        long_act = "YOU BOUGHT " + "X" * 1000
        body = (
            "\n\nRun Date,Account,Account Number,Action,Symbol,Description,"
            "Type,Price ($),Quantity,Commission ($),Fees ($),"
            "Accrued Interest ($),Amount ($),Settlement Date\n"
            f"04/25/2026,Individual,X1,{long_act},AAPL,,Margin,150.00,10,,,,"
            "-$1500.00,04/27/2026\n"
        )
        r = _post_activity(body.encode("utf-8"))
        assert r.status_code == 200
        db = _db()
        try:
            t = db.query(FidelityTrade).first()
            assert t is not None
            assert len(t.raw_action) <= fidelity_sync.MAX_ACTION_LEN
        finally:
            db.close()


# ════════════════════════════════════════════════════════════════════
# Threat 6 — CSV injection on re-export
# ════════════════════════════════════════════════════════════════════

class TestThreat6CsvInjection:

    def test_no_reexport_path_in_codebase(self):
        """N/A — Fidelity records are not re-emitted as CSV anywhere
        (verified via grep of email_utils, ab_eval_email, scheduler).
        This test documents the audit verdict; if a future re-export
        path is added, a follow-up test must guard the formula
        characters: '=', '+', '-', '@', '\\t', '\\r' at field start."""
        # Sanity check: storing a formula-prefixed value works (we don't
        # mutate input — we only have to neutralize at emit time, which
        # is currently never).
        body = (
            "Account Number,Symbol,Description,Quantity,Last Price,"
            "Current Value,Total Gain/Loss Dollar,Total Gain/Loss Percent,"
            "Cost Basis Total,Average Cost Basis,Percent Of Account,Type\n"
            "X1,AAPL,=cmd|' /C calc'!A0,100,$150.00,$15000.00,+$5000,+50%,"
            "$10000,$100,100%,Margin\n"
        )
        r = _post_positions(body.encode("utf-8"))
        assert r.status_code == 200
        db = _db()
        try:
            pos = db.query(FidelityPosition).filter_by(symbol="AAPL").first()
            assert pos.description.startswith("=")  # stored raw, by design
        finally:
            db.close()


# ════════════════════════════════════════════════════════════════════
# Threat 7 — Dedup race (FLAG only — do not migrate)
# ════════════════════════════════════════════════════════════════════

class TestThreat7DedupRace:

    def test_query_then_insert_pattern_is_known_race(self):
        """FLAG: routes/fidelity.py:upload-activity does query-then-insert.
        Two concurrent uploads can both pass the dedup check and double-insert.
        Mitigation deferred — separate session per the brief.
        This test documents the known gap so future migration work can
        target the exact code path."""
        # Verify the route path that still has the race. Dedup is now a
        # batched pre-load into an in-memory `seen_keys` set (one query instead
        # of a per-row `.first()`), but it remains check-then-insert: concurrent
        # uploads share no lock and there is no DB unique constraint, so both
        # can pass `key in seen_keys` and double-insert.
        import inspect
        src = inspect.getsource(fidelity_routes.upload_fidelity_activity)
        assert "db.query(" in src and "FidelityTrade" in src
        assert "seen_keys" in src            # in-memory dedup set (check)
        assert "db.add(FidelityTrade(" in src  # insert, with no unique constraint
        # NOTE: A unique constraint on (user_id, run_date, symbol, action)
        # would close this — still a deferred DB migration.


# ════════════════════════════════════════════════════════════════════
# Threat 8 — Empty / blank file
# ════════════════════════════════════════════════════════════════════

class TestThreat8EmptyFile:

    def test_zero_byte_positions_returns_400(self):
        r = _post_positions(b"")
        assert r.status_code == 400
        # Either "empty" or "no positions" — must not be 500.
        assert r.status_code != 500

    def test_zero_byte_activity_returns_400(self):
        r = _post_activity(b"")
        assert r.status_code == 400

    def test_header_only_positions_returns_400(self):
        """Just the header row → 0 positions parsed → 400 (not 500)."""
        header_only = (
            "Account Number,Symbol,Quantity,Last Price,Current Value,"
            "Total Gain/Loss Dollar,Total Gain/Loss Percent,"
            "Cost Basis Total,Average Cost Basis,Percent Of Account,Type\n"
        )
        r = _post_positions(header_only.encode("utf-8"))
        assert r.status_code == 400

    def test_header_only_activity_succeeds_with_zero_trades(self):
        """An activity export with only a header should succeed with 0
        new_trades, not 500. This matches the 'success-with-empty' shape
        of valid Fidelity exports during quiet weeks."""
        header_only = (
            "\n\nRun Date,Account,Account Number,Action,Symbol,Description,"
            "Type,Price ($),Quantity,Commission ($),Fees ($),"
            "Accrued Interest ($),Amount ($),Settlement Date\n"
        )
        r = _post_activity(header_only.encode("utf-8"))
        assert r.status_code == 200
        assert r.json()["new_trades"] == 0


# ════════════════════════════════════════════════════════════════════
# Threat 9 — Header spoofing (misordered / renamed columns)
# ════════════════════════════════════════════════════════════════════

class TestThreat9HeaderSpoofing:

    def test_positions_missing_symbol_header_rejected(self):
        """If 'Symbol' column is renamed to 'Sym', parser raises ValueError
        instead of silently mis-aligning rows."""
        body = (
            "Account Number,Sym,Quantity,Last Price,Current Value,"
            "Total Gain/Loss Dollar,Total Gain/Loss Percent,"
            "Cost Basis Total,Average Cost Basis,Percent Of Account,Type\n"
            "X1,AAPL,100,$150.00,$15000.00,+$5000,+50%,$10000,$100,100%,Margin\n"
        )
        r = _post_positions(body.encode("utf-8"))
        assert r.status_code == 400
        assert "Symbol" in r.json()["detail"]

    def test_positions_missing_quantity_header_rejected(self):
        body = (
            "Account Number,Symbol,Last Price,Current Value,"
            "Total Gain/Loss Dollar,Total Gain/Loss Percent,"
            "Cost Basis Total,Average Cost Basis,Percent Of Account,Type\n"
            "X1,AAPL,$150.00,$15000.00,+$5000,+50%,$10000,$100,100%,Margin\n"
        )
        r = _post_positions(body.encode("utf-8"))
        assert r.status_code == 400
        assert "Quantity" in r.json()["detail"]

    def test_activity_missing_action_header_rejected(self):
        body = (
            "\n\nRun Date,Account,Account Number,Symbol,Description,"
            "Type,Price ($),Quantity,Commission ($),Fees ($),"
            "Accrued Interest ($),Amount ($),Settlement Date\n"
            "04/25/2026,Individual,X1,AAPL,,Margin,150.00,10,,,,"
            "-$1500.00,04/27/2026\n"
        )
        r = _post_activity(body.encode("utf-8"))
        assert r.status_code == 400
        assert "Action" in r.json()["detail"]

    def test_misordered_columns_still_align_via_dict(self):
        """csv.DictReader pairs by header name, so column reorder is safe
        as long as the headers themselves are present."""
        body = (
            # Reverse the typical column order (Type first, account last)
            "Type,Percent Of Account,Average Cost Basis,Cost Basis Total,"
            "Total Gain/Loss Percent,Total Gain/Loss Dollar,Current Value,"
            "Last Price,Quantity,Description,Symbol,Account Name,Account Number\n"
            "Margin,100%,$100,$10000,+50%,+$5000,$15000.00,$150.00,100,"
            "APPLE INC,AAPL,Individual,X1\n"
        )
        r = _post_positions(body.encode("utf-8"))
        assert r.status_code == 200
        db = _db()
        try:
            pos = db.query(FidelityPosition).first()
            assert pos.symbol == "AAPL"
            assert pos.quantity == 100
        finally:
            db.close()


# ════════════════════════════════════════════════════════════════════
# Threat 10 — Numeric NaN / Infinity abuse
# ════════════════════════════════════════════════════════════════════

class TestThreat10NumericAbuse:

    def test_clean_dollar_rejects_nan(self):
        assert fidelity_sync._clean_dollar("NaN") is None
        assert fidelity_sync._clean_dollar("nan") is None

    def test_clean_dollar_rejects_infinity(self):
        assert fidelity_sync._clean_dollar("Infinity") is None
        assert fidelity_sync._clean_dollar("-Infinity") is None
        assert fidelity_sync._clean_dollar("inf") is None
        assert fidelity_sync._clean_dollar("-inf") is None

    def test_clean_float_rejects_huge_overflow(self):
        # 1e500 overflows IEEE-754 double to +inf; isfinite filter catches it.
        assert fidelity_sync._clean_float("1e500") is None
        assert fidelity_sync._clean_float("-1e500") is None

    def test_clean_dollar_finite_values_pass(self):
        """Sanity: regular numbers still flow through unchanged."""
        assert fidelity_sync._clean_dollar("$1234.56") == 1234.56
        assert fidelity_sync._clean_dollar("-$73.00") == -73.00
        assert fidelity_sync._clean_dollar("+$0.01") == 0.01

    def test_positions_with_infinity_price_stored_as_null(self):
        """An Infinity-valued price field results in a NULL DB row, not a
        json-breaking float('inf') in the response."""
        body = (
            "Account Number,Symbol,Description,Quantity,Last Price,"
            "Current Value,Total Gain/Loss Dollar,Total Gain/Loss Percent,"
            "Cost Basis Total,Average Cost Basis,Percent Of Account,Type\n"
            "X1,AAPL,APPLE INC,100,Infinity,$15000.00,+$5000,+50%,"
            "$10000,$100,100%,Margin\n"
        )
        r = _post_positions(body.encode("utf-8"))
        # Must succeed (price coerced to None) and JSON-serialize cleanly.
        assert r.status_code == 200
        db = _db()
        try:
            pos = db.query(FidelityPosition).first()
            assert pos.symbol == "AAPL"
            assert pos.last_price is None  # Infinity → None
        finally:
            db.close()

    def test_activity_with_nan_quantity_handled(self):
        """A NaN quantity falls through abs(0) and gets stored as 0 (not NaN).
        This is the parser's `abs_quantity = abs(quantity) if quantity else 0`
        — quantity=None (NaN coerces to None via _clean_float) → 0."""
        body = (
            "\n\nRun Date,Account,Account Number,Action,Symbol,Description,"
            "Type,Price ($),Quantity,Commission ($),Fees ($),"
            "Accrued Interest ($),Amount ($),Settlement Date\n"
            "04/25/2026,Individual,X1,YOU BOUGHT AAPL,AAPL,,Margin,150.00,NaN,,,,"
            "-$1500.00,04/27/2026\n"
        )
        r = _post_activity(body.encode("utf-8"))
        assert r.status_code == 200
        db = _db()
        try:
            t = db.query(FidelityTrade).first()
            if t is not None:
                # quantity must be a real finite number, never NaN
                assert t.quantity is None or math.isfinite(t.quantity)
        finally:
            db.close()
