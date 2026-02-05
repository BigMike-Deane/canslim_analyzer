# Archive

This folder contains legacy code that is no longer actively used but preserved for reference.

## Files

### `email_report_legacy.py` (archived Feb 5, 2026)

**Original**: `email_report.py`

**Reason**: This was a standalone CLI script for sending daily CANSLIM email reports. It was from the first iteration of the project before the web application existed.

**Issues**:
- Imported from root `main.py` (not `backend/main.py`) causing `ModuleNotFoundError` in container
- Daily report functionality was never integrated with the web app
- Contained separate stock analysis logic that duplicated the web API

**What was preserved**:
- `send_watchlist_alert_email()` and `send_email()` functions were extracted to `backend/email_utils.py`
- These functions ARE actively used by `backend/scheduler.py` for watchlist price/score alerts

**If you need daily email reports in the future**:
Consider building a new integration that uses the existing `/api/` endpoints rather than running duplicate analysis.
