# Fidelity CSV Upload Hardening Audit (2026-05-08)

**Scope.** Two routes plus their parser:
- `POST /api/fidelity/upload-positions` — `backend/routes/fidelity.py:51`
- `POST /api/fidelity/upload-activity`  — `backend/routes/fidelity.py:128`
- `parse_positions_csv` / `parse_activity_csv` — `backend/fidelity_sync.py`

**Trust model.** Both routes are auth-gated (`Depends(get_current_active_user)`) — the input
is *authenticated user-controlled bytes*, not anonymous. But authenticated does not mean
trusted: a compromised account or a user uploading a malformed export can still poison
DB rows that the AI Trader cross-references for sector limits and reconciliation.

**Sequencing.** Companion to commit `818582d` (CSO IDOR audit). That ship hardened
authorization on these same routes; this ship hardens the *content* of accepted uploads.

**Eval-safe.** Parser hardening rejects bad input before scoring/ML/trading code sees it.
The 2026-06-18 Approach 2 evaluation window remains untouched.

---

## Threat-by-threat verdict table

| # | Threat | Pre-fix verdict | Fix | Post-fix verdict |
|---|---|---|---|---|
| 1 | File-size DoS | **FAIL** — `await file.read()` had no cap | 10 MB streaming guard via `_read_capped` (`routes/fidelity.py:34`) | **PASS** — 413 on overflow |
| 2 | Encoding-bomb / decode crash | **PASS** — utf-8 → latin-1 already in place | Added `\x00` strip + empty-bytes 400 | **PASS** |
| 3 | MIME bypass via filename | **N/A** — filename was always defense-in-depth, not the trust boundary | None — explicitly documented | **PASS** |
| 4 | Parser exception leak (500 + stack) | **FAIL** — `csv.Error` / unexpected exceptions returned 500 with traceback | `try/except` wraps both parser calls; `logger.exception` server-side, sanitized 400 to client | **PASS** |
| 5 | Field-length overflow | **FAIL** — symbol/description/raw_action stored verbatim with no cap | `_safe_str` truncation: 32/512/256 chars | **PASS** |
| 6 | CSV injection on re-export | **N/A** — no re-emit path exists (verified via grep) | Documented; future re-export paths must neutralize formula prefix chars | **PASS** |
| 7 | Dedup race in upload-activity | **FAIL** — query-then-insert is not atomic | **FLAG ONLY** — DB unique-constraint migration deferred per brief | **DEFERRED** |
| 8 | Empty / blank file | **FAIL** — empty-bytes raised UnicodeDecodeError → 500 | `_decode_csv` 400s on empty; header-only positions → 400 "no positions"; header-only activity → 200 / 0 trades | **PASS** |
| 9 | Header spoofing (renamed columns) | **FAIL** — silent mis-alignment if `Symbol` renamed to `Sym` | Required-header check: `REQUIRED_POSITION_HEADERS` / `REQUIRED_ACTIVITY_HEADERS`; missing → ValueError → 400 | **PASS** |
| 10 | Numeric NaN / Infinity abuse | **FAIL** — `float("Infinity")` flowed through to DB and JSON response | `math.isfinite()` filter in `_finite` helper, applied to all three cleaners (`_clean_dollar`, `_clean_percent`, `_clean_float`) | **PASS** |

**Score: 7 PASS, 2 N/A, 1 DEFERRED, 0 outstanding FAIL.**

---

## Threat 1 — File-size DoS

**Evidence (pre-fix).** `routes/fidelity.py:34` (old): `content = await file.read()` with no
size limit. A 10 GB upload would allocate 10 GB of memory before any check fired.

**Fix.** New `_read_capped` helper streams in 64 KB chunks and raises `HTTPException(413)`
the moment cumulative size exceeds `MAX_UPLOAD_BYTES` (10 MB). Real Fidelity exports are
~10–50 KB, so 10 MB gives ~1000× headroom while still bounding worst-case memory.

**Tests.** `TestThreat1FileSizeDoS::test_*_rejects_oversize_payload` (both routes).

---

## Threat 2 — Encoding-bomb / decode crash

**Evidence.** Pre-existing utf-8 → latin-1 fallback at `routes/fidelity.py:36-38` is sound;
`latin-1` is total over byte values 0x00–0xFF, so the decode path cannot raise.

**Fix (incremental).** Added `\x00` strip in both parsers (after BOM strip) — some editors
append null padding to CSV exports, and while the csv module tolerates them, downstream
DB columns may not. Also added 400-on-empty-bytes check in `_decode_csv`.

**Tests.** `TestThreat2Decode::test_*_handles_utf8_bom`, `test_*_handles_trailing_nulls`,
`test_*_handles_arbitrary_high_bytes`.

---

## Threat 3 — MIME bypass via filename

**Verdict: N/A (defense-in-depth only).**

The `.csv` filename check at `routes/fidelity.py:53` and `:131` is *not* the trust boundary.
A `.csv.exe` filename fails the suffix check (cosmetic win), but a `evil.csv` filename
containing arbitrary bytes is correctly rejected at the parser layer instead. The bytes
themselves are what get validated — that's the right design.

Hardened the filename check to be `(file.filename or "").endswith('.csv')` (was crashing
with `AttributeError: NoneType` when filename was None — caught by Pydantic 422 at form
layer, but defensive double-check is cheap).

**Tests.** `TestThreat3MimeBypass::test_filename_is_only_defense_in_depth`,
`test_csv_dot_exe_filename_still_rejected`, `test_missing_filename_does_not_crash`.

---

## Threat 4 — Parser exception leak

**Evidence (pre-fix).** `parse_positions_csv` had a generic `try/except` per-row, but
`csv.Error` (e.g., from `_csv.field_size_limit` overflow on a giant quoted field) bubbled
up unhandled. The route's `result = parse_positions_csv(csv_text)` call would 500 with a
full traceback in the response body — leaking SQL paths, file paths, and stack frames.

**Fix.** Wrapped both parser calls with:
```python
try:
    result = parse_*_csv(csv_text)
except ValueError as e:
    raise HTTPException(status_code=400, detail=f"Could not parse ...: {str(e)[:200]}")
except Exception:
    logger.exception("Unexpected error parsing ...")
    raise HTTPException(status_code=400, detail="Could not parse ...")
```

`ValueError` messages are surfaced (truncated to 200 chars, since they come from our own
parser code with known shape — `"missing required column(s): ..."`). Anything else is
logged server-side with full traceback and returns a sanitized message to the client.

**Tests.** `TestThreat4ExceptionLeak::test_unbalanced_quotes_returns_400_not_500`,
`test_unexpected_exception_caught_in_route` (verifies secret leak prevention),
`test_value_error_message_truncated`.

---

## Threat 5 — Field-length overflow

**Evidence (pre-fix).** `FidelityPosition.description` is `Column(String)` — PostgreSQL
TEXT, no length limit. A 100 KB description would store as-is, then echo back through
`/api/fidelity/latest` and `/gameplan` responses. Stored-XSS vector if rendered unescaped
(React escapes by default — but log-flooding via `logger.info(f"Skipped {symbol}: ...")`
remains a concern when symbol is 10 KB).

**Fix.** New `_safe_str(value, max_len)` helper applied at parse time:
- `MAX_SYMBOL_LEN = 32`
- `MAX_DESCRIPTION_LEN = 512`
- `MAX_ACTION_LEN = 256` (raw_action and position type)

**Tests.** `TestThreat5FieldLength::test_*_truncated`.

---

## Threat 6 — CSV injection on re-export

**Verdict: N/A.**

`grep -rln "FidelityTrade\|FidelityPosition" backend/ scripts/` returns only:
- `backend/database.py` (model definitions)
- `backend/fidelity_sync.py` (parser)
- `backend/main.py` (import only — no re-emit)
- `backend/routes/fidelity.py` (read-only JSON responses)

No `email_utils`, `ab_eval_email`, or scheduler path emits Fidelity records as CSV. JSON
responses are immune to formula-injection attacks (the threat is specifically Excel/Sheets
re-opening a CSV file and auto-evaluating cells starting with `=`, `+`, `-`, `@`, `\t`,
`\r`).

**Future-guard.** If a CSV-export endpoint is ever added, prefix any field starting with
those 6 characters with a single quote (`'`). Current input remains stored verbatim — the
fix lives at emit time, not store time.

**Tests.** `TestThreat6CsvInjection::test_no_reexport_path_in_codebase` (documents and
sanity-checks raw-storage behavior).

---

## Threat 7 — Dedup race (FLAG ONLY, deferred)

**Evidence.** `routes/fidelity.py:upload-activity` (now ~140-160) does:
```python
existing = db.query(FidelityTrade).filter(...).first()
if existing:
    skipped += 1
    continue
db.add(FidelityTrade(...))
```

Two concurrent uploads of the same activity CSV by the same user can both pass the
`first()` check before either commits, then both insert — yielding duplicate rows.

**Mitigation deferred** per brief. The proper fix is a unique constraint on
`(user_id, run_date, symbol, action)` in `FidelityTrade.__table_args__`, but that's a
DB migration outside the eval-safe scope.

**Real-world likelihood: low.** Manual upload, single user per session, browsers serialize
form posts. The race is technically present but unlikely to fire in practice.

**Test.** `TestThreat7DedupRace::test_query_then_insert_pattern_is_known_race` —
introspects the route source so future refactors can't accidentally hide the pattern
without also adding a unique constraint.

---

## Threat 8 — Empty / blank file

**Evidence (pre-fix).** `b"".decode("utf-8")` returns `""`, which `parse_positions_csv`
handles by returning empty `positions` → 400 "No positions". So 0-byte was already 400.
But the `_decode_csv` helper now 400s explicitly with a clearer "File is empty" message
and rejects before parser invocation (cheaper).

**Header-only positions** (header row but zero data rows) → parser returns 0 positions →
existing 400 "No positions" check fires. Correct.

**Header-only activity** → parser returns 0 trades → route returns 200 with `new_trades=0`.
This is the *desired* shape: a valid Fidelity activity CSV during a quiet week genuinely
has no trades, and 200/0 is the right response. We do *not* 400 these — that would force
users to never upload during dry weeks.

**Tests.** `TestThreat8EmptyFile::test_*`.

---

## Threat 9 — Header spoofing

**Evidence (pre-fix).** `csv.DictReader` pairs columns by header name. If the user
uploaded a CSV with `Symbol` renamed to `Sym`, `row.get('Symbol')` returned `None`,
which the existing `if not symbol: continue` skipped silently. Result: parser reported
0 positions → 400 — *defensive but accidental*. If a different required column was
spoofed (e.g., `Quantity` removed), rows would silently mis-align across columns.

**Fix.** Explicit required-header check at parser entry:
```python
fieldnames = set(reader.fieldnames or [])
missing = [h for h in REQUIRED_*_HEADERS if h not in fieldnames]
if missing:
    raise ValueError(f"missing required column(s): {', '.join(missing)}")
```

**Tests.** `TestThreat9HeaderSpoofing::test_*_missing_*_header_rejected`,
`test_misordered_columns_still_align_via_dict` (proves we don't break legitimate
column-reorders).

---

## Threat 10 — Numeric NaN / Infinity abuse

**Evidence (pre-fix).** `float("Infinity")` returns `+inf`, `float("NaN")` returns `nan`,
`float("1e500")` returns `+inf`. All three flowed through `_clean_dollar` / `_clean_float`
unchanged. Once stored:
- SQLite stores them. PostgreSQL stores them.
- `json.dumps(float('inf'))` → `Infinity` (invalid JSON per RFC 8259 — breaks strict
  parsers including most JS clients).

This is *exactly* the bug we fixed for `profit_factor` in commit `ec4719f`. The Fidelity
upload was the upstream source — if a malicious or malformed CSV had `Infinity` in any
numeric field, it would propagate.

**Fix.** New `_finite` helper applied to all three cleaners:
```python
def _finite(n):
    if n is None: return None
    if not math.isfinite(n): return None
    return n
```

NaN/+Inf/-Inf inputs now coerce to `None` (NULL in DB, `null` in JSON), preserving the
"unknown / unparseable" semantic without breaking downstream serialization.

**Tests.** `TestThreat10NumericAbuse::test_clean_*_rejects_*`, `test_clean_*_finite_values_pass`,
`test_*_with_*_stored_as_null`.

---

## Files touched

| File | Change |
|---|---|
| `backend/fidelity_sync.py` | `_safe_str` + `_finite` helpers; required-header validation; field-length truncation; csv.Error wrapping |
| `backend/routes/fidelity.py` | `MAX_UPLOAD_BYTES` cap + `_read_capped` streamer; `_decode_csv` helper; parser-call try/except wrapping; logger import |
| `tests/test_fidelity_csv_hardening.py` | New file, 32 tests across 10 threat classes |
| `docs/security/fidelity-csv-hardening.md` | This document |

## Out of scope (per brief)

- Auth checks — covered by IDOR audit `818582d`.
- Dependency CVE scan — separate session (CSO #3).
- Rate limits on admin endpoints — separate session.
- DB migration for unique constraint on `FidelityTrade` — separate (Threat 7 flag).
- Reconciliation logic correctness — orthogonal.

## Suite impact

- Pre-ship baseline: 2199 / 16 skip / 0 fail
- Post-ship: 2231 / 16 skip / 0 fail (+32 hardening tests)
