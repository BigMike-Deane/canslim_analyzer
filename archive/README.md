# Archive

This folder contains legacy code that is no longer actively used but preserved for reference.

## Files

### `email_report_legacy.py` (archived Feb 5, 2026)
**Original**: `email_report.py`

Standalone CLI script for sending daily CANSLIM email reports. From v1 of the project before the web application existed. Imported from root `main.py` causing `ModuleNotFoundError` in container.

**Preserved functions**: `send_watchlist_alert_email()` and `send_email()` were extracted to `backend/email_utils.py`

---

### `main.py` (archived Feb 5, 2026)
**Original**: `main.py` (root)

CLI version of the CANSLIM analyzer with argparse interface. Replaced by `backend/main.py` (FastAPI web app). Was only imported by `email_report_legacy.py`.

---

### `score_history.py` (archived Feb 5, 2026)

Module for tracking score changes over time via JSON files. Was used by the legacy email report for "biggest movers" section. The web app uses the `StockScore` database table instead.

---

### `portfolio_analyzer.py` (archived Feb 5, 2026)

Portfolio analysis module that loaded positions from CSV files and generated recommendations. Was used by the legacy email report. The web app has its own portfolio management via database (`PortfolioPosition` model) and `/api/portfolio/*` endpoints.

---

### `portfolio_manager.py` (archived Feb 5, 2026)

Portfolio management utilities. Never imported by any active code.

---

### `fix_missing_targets.py` (archived Feb 5, 2026)

One-time maintenance script to fix missing analyst price targets for major stocks. Run manually after scans if needed. Not part of the web application.

---

### `send_demo_email.py` (archived Feb 5, 2026)

Demo script for testing email format with sample data. Used during development to verify email templates.

---

## Archived Documentation (Feb 5, 2026)

These docs are superseded by `CLAUDE.md` which contains comprehensive, up-to-date project context.

### `CURRENT_STATUS.md`
Status updates from Jan 29, 2026. Now outdated.

### `IMPLEMENTATION_SUMMARY.md`
Describes the initial config system, Redis cache, and unit test implementation from Jan 23.

### `ASYNC_IMPLEMENTATION.md`
Guide for the async scanner implementation from Jan 23. Performance numbers and patterns are now in CLAUDE.md.

### `README_TESTING.md`
Testing guide from Jan 23. Test commands and patterns are now in CLAUDE.md.

---

## If You Need These Features

- **Daily email reports**: Build a new integration using existing `/api/` endpoints rather than running duplicate analysis
- **Score history/trends**: Use `GET /api/stocks/{ticker}` which includes `score_history` from the database
- **Portfolio analysis**: Use `/api/portfolio/gameplan` endpoint
- **Maintenance scripts**: Can still be run manually from this archive folder if needed
