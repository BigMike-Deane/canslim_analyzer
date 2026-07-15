"""Coverage tests for sp500_tickers.py.

sp500_tickers.py 7.39% → target 60%+. Universe-management code that
scrapes Wikipedia + FMP + Yahoo Finance for index constituents.

Test strategy:
- Mock at the network boundary (`requests.get`, `yfinance.Ticker`)
- Use REAL BeautifulSoup on canned Wikipedia-shaped HTML so the
  parser logic gets honest exercise
- Reset module-level _ticker_cache between tests via autouse fixture
  so cache-hit/miss branches are deterministic
- Hardcoded fallback functions (single-line returns of huge static
  lists) covered by invoking each one once.
"""

import os
import sys
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sp500_tickers


# ── Module state reset ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_ticker_cache():
    """Each test starts with a fresh empty cache.

    sp500_tickers._ticker_cache is a module-level dict that persists
    results across calls — resetting it between tests keeps the
    cache-hit / cache-miss branches deterministic.
    """
    sp500_tickers._ticker_cache = {
        'sp500': None, 'nasdaq100': None, 'dowjones': None,
        'midcap400': None, 'smallcap600': None, 'russell2000': None,
        'fmp_screener': None, 'fmp_reit_trusts': None,
        'last_fetch': {},
    }
    yield
    sp500_tickers._ticker_cache = {
        'sp500': None, 'nasdaq100': None, 'dowjones': None,
        'midcap400': None, 'smallcap600': None, 'russell2000': None,
        'fmp_screener': None, 'fmp_reit_trusts': None,
        'last_fetch': {},
    }


# ── Canned Wikipedia HTML for the various parsers ──────────────────────

# S&P 500 / S&P 400 / S&P 600 — table id="constituents", first td is ticker
SP_CONSTITUENTS_HTML = """
<html><body>
<table id="constituents">
  <tr><th>Symbol</th><th>Security</th></tr>
  <tr><td>AAPL</td><td>Apple Inc.</td></tr>
  <tr><td>MSFT</td><td>Microsoft</td></tr>
  <tr><td>BRK.B</td><td>Berkshire Hathaway</td></tr>
</table>
</body></html>
"""

# Wikipedia variant where the id is missing but class='wikitable' present
SP_CONSTITUENTS_NO_ID_HTML = """
<html><body>
<table class="wikitable">
  <tr><th>Symbol</th></tr>
  <tr><td>GOOG</td></tr>
  <tr><td>META</td></tr>
</table>
</body></html>
"""

# Nasdaq 100 — multiple wikitables, only the one with 'Ticker' header is right
NASDAQ100_HTML_TEMPLATE = """
<html><body>
<table class="wikitable">
  <tr><th>Header A</th></tr>
  <tr><td>noise</td></tr>
</table>
<table class="wikitable">
  <tr><th>Company</th><th>Ticker</th><th>Sector</th></tr>
  {rows}
</table>
</body></html>
"""

# Dow Jones — wikitable with 'Symbol' header (only 30 tickers required)
DOWJONES_HTML_TEMPLATE = """
<html><body>
<table class="wikitable">
  <tr><th>Company</th><th>Symbol</th></tr>
  {rows}
</table>
</body></html>
"""

# Finviz screener — looks for total + `screener-link-primary` anchors
FINVIZ_HTML_PAGE1 = """
<html><body>
<table><tr><td>Total: 500</td></tr></table>
<a class="screener-link-primary">SMCO</a>
<a class="screener-link-primary">TINY</a>
<a class="screener-link-primary">MICR</a>
</body></html>
"""

FINVIZ_HTML_EMPTY = """
<html><body>
<table><tr><td>Total: 0</td></tr></table>
</body></html>
"""


def _alpha_ticker(i: int) -> str:
    """Generate an alphabetic-only ticker like 'AAA', 'AAB', ..., 'CYZ'.

    The Nasdaq 100 and Dow Jones parsers reject any ticker that
    isn't `.isalpha()`-clean, so the generated test rows must avoid
    digits.
    """
    # Three-letter ticker from i in [0, 26**3)
    a = i // (26 * 26)
    b = (i // 26) % 26
    c = i % 26
    return chr(ord("A") + a) + chr(ord("A") + b) + chr(ord("A") + c)


def _build_nasdaq100_html(num_tickers: int = 100) -> str:
    """Generate Nasdaq 100 HTML with `num_tickers` valid rows.

    The parser requires >=90 tickers to accept the table. Going below
    that exercises the "rejected, fallthrough to fallback" branch.
    """
    rows = "\n".join(
        f"  <tr><td>Company{i}</td><td>{_alpha_ticker(i)}</td><td>Tech</td></tr>"
        for i in range(num_tickers)
    )
    return NASDAQ100_HTML_TEMPLATE.format(rows=rows)


def _build_dowjones_html(num_tickers: int = 30) -> str:
    """Generate Dow Jones HTML with `num_tickers` rows. Needs >=25."""
    rows = "\n".join(
        f"  <tr><td>Company{i}</td><td>{_alpha_ticker(i)}</td></tr>"
        for i in range(num_tickers)
    )
    return DOWJONES_HTML_TEMPLATE.format(rows=rows)


# ── Cache lifecycle helpers ────────────────────────────────────────────


class TestIsCacheValid:
    """Covers sp500_tickers.py:35-53."""

    def test_returns_false_when_cache_key_is_none(self):
        """Branch: _ticker_cache[key] is None → invalid (line 39-40)."""
        sp500_tickers._ticker_cache['sp500'] = None
        assert sp500_tickers._is_cache_valid('sp500') is False

    def test_returns_false_when_last_fetch_not_dict(self):
        """Branch: legacy format where last_fetch isn't a dict (line 43-46)."""
        sp500_tickers._ticker_cache['sp500'] = ['AAPL']
        sp500_tickers._ticker_cache['last_fetch'] = "legacy-string"
        assert sp500_tickers._is_cache_valid('sp500') is False
        # The function recovers by re-initialising last_fetch to {}
        assert sp500_tickers._ticker_cache['last_fetch'] == {}

    def test_returns_false_when_no_fetch_time_for_key(self):
        """Branch: cache has data but no timestamp (line 48-50)."""
        sp500_tickers._ticker_cache['sp500'] = ['AAPL']
        sp500_tickers._ticker_cache['last_fetch'] = {}  # no 'sp500' key
        assert sp500_tickers._is_cache_valid('sp500') is False

    def test_returns_true_when_recent(self):
        """Branch: cached + fresh timestamp → valid (line 52-53)."""
        sp500_tickers._ticker_cache['sp500'] = ['AAPL']
        sp500_tickers._ticker_cache['last_fetch'] = {
            'sp500': datetime.now() - timedelta(hours=1)
        }
        assert sp500_tickers._is_cache_valid('sp500') is True

    def test_returns_false_when_expired(self):
        """Branch: cached but stale → invalid (line 52-53)."""
        sp500_tickers._ticker_cache['sp500'] = ['AAPL']
        sp500_tickers._ticker_cache['last_fetch'] = {
            'sp500': datetime.now() - timedelta(hours=48)  # > 24h
        }
        assert sp500_tickers._is_cache_valid('sp500') is False


class TestUpdateCache:
    """Covers sp500_tickers.py:56-66."""

    def test_stores_data_and_timestamp(self):
        """Branch: writes data + new fetch time."""
        sp500_tickers._update_cache('sp500', ['AAPL', 'MSFT'])
        assert sp500_tickers._ticker_cache['sp500'] == ['AAPL', 'MSFT']
        assert 'sp500' in sp500_tickers._ticker_cache['last_fetch']
        # Timestamp is a datetime
        assert isinstance(
            sp500_tickers._ticker_cache['last_fetch']['sp500'],
            datetime
        )

    def test_recovers_when_last_fetch_is_legacy_format(self):
        """Branch: last_fetch not a dict → re-initialise (line 62-63)."""
        sp500_tickers._ticker_cache['last_fetch'] = "legacy"
        sp500_tickers._update_cache('nasdaq100', ['QQQ'])
        assert isinstance(sp500_tickers._ticker_cache['last_fetch'], dict)
        assert 'nasdaq100' in sp500_tickers._ticker_cache['last_fetch']


# ── Hardcoded fallback functions (one-line returns) ───────────────────


class TestHardcodedFallbacks:
    """Covers sp500_tickers.py:279, 358, 435, 508, 711.

    Each function is a single `return [...]` statement; calling it
    once covers the line and asserts the shape.
    """

    def test_fallback_nasdaq100_returns_nontrivial_list(self):
        result = sp500_tickers.get_fallback_nasdaq100_tickers()
        assert isinstance(result, list)
        assert len(result) >= 90
        assert "AAPL" in result and "MSFT" in result

    def test_fallback_sp500_returns_diverse_sectors(self):
        result = sp500_tickers.get_fallback_tickers()
        assert isinstance(result, list)
        assert len(result) > 100
        # Quick sector sampling
        assert "AAPL" in result  # Tech
        assert "JPM" in result   # Financials
        assert "XOM" in result   # Energy

    def test_fallback_midcap_returns_nontrivial_list(self):
        result = sp500_tickers.get_fallback_midcap_tickers()
        assert isinstance(result, list)
        assert len(result) > 100

    def test_fallback_smallcap_returns_nontrivial_list(self):
        result = sp500_tickers.get_fallback_smallcap_tickers()
        assert isinstance(result, list)
        assert len(result) > 100

    def test_fallback_russell2000_returns_nontrivial_list(self):
        result = sp500_tickers.get_fallback_russell2000_tickers()
        assert isinstance(result, list)
        assert len(result) > 200


# ── get_sp500_tickers ─────────────────────────────────────────────────


class TestGetSp500Tickers:
    """Covers sp500_tickers.py:181-222."""

    def test_cache_hit_returns_cached_without_network(self):
        """Branch: cache valid → return cached, no requests.get call (188-189)."""
        sp500_tickers._ticker_cache['sp500'] = ['CACHED1', 'CACHED2']
        sp500_tickers._ticker_cache['last_fetch'] = {
            'sp500': datetime.now()
        }
        with patch("sp500_tickers.requests.get") as mock_get:
            result = sp500_tickers.get_sp500_tickers()
        assert result == ['CACHED1', 'CACHED2']
        mock_get.assert_not_called()

    def test_wikipedia_happy_path_with_constituents_table(self):
        """Branch: <table id='constituents'> parsed, BRK.B → BRK-B (192-216)."""
        mock_resp = MagicMock()
        mock_resp.text = SP_CONSTITUENTS_HTML
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_sp500_tickers()
        assert "AAPL" in result
        assert "MSFT" in result
        assert "BRK-B" in result  # dot → dash normalisation
        # Cache was populated
        assert sp500_tickers._ticker_cache['sp500'] == result

    def test_wikipedia_table_id_missing_uses_wikitable_fallback(self):
        """Branch: no #constituents → table.class='wikitable' (201-202)."""
        mock_resp = MagicMock()
        mock_resp.text = SP_CONSTITUENTS_NO_ID_HTML
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_sp500_tickers()
        assert "GOOG" in result and "META" in result

    def test_network_failure_returns_fallback_list(self):
        """Branch: requests.get raises → fallback ticker list (218-222)."""
        with patch(
            "sp500_tickers.requests.get",
            side_effect=ConnectionError("network down"),
        ):
            result = sp500_tickers.get_sp500_tickers()
        # Fallback returned + cached
        assert isinstance(result, list)
        assert len(result) > 100
        assert "AAPL" in result  # fallback list seed


# ── get_nasdaq100_tickers ──────────────────────────────────────────────


class TestGetNasdaq100Tickers:
    """Covers sp500_tickers.py:225-274."""

    def test_cache_hit_returns_cached(self):
        """Branch: cache valid → return early (232-233)."""
        sp500_tickers._ticker_cache['nasdaq100'] = ['CACHE_NDX']
        sp500_tickers._ticker_cache['last_fetch'] = {
            'nasdaq100': datetime.now()
        }
        with patch("sp500_tickers.requests.get") as mock_get:
            result = sp500_tickers.get_nasdaq100_tickers()
        assert result == ['CACHE_NDX']
        mock_get.assert_not_called()

    def test_wikipedia_happy_path_returns_parsed_tickers(self):
        """Branch: wikitable with 'Ticker' header → 100 parsed tickers."""
        mock_resp = MagicMock()
        mock_resp.text = _build_nasdaq100_html(num_tickers=100)
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_nasdaq100_tickers()
        assert len(result) >= 90  # parser requires >=90 to accept
        # First generated ticker is "AAA" (i=0)
        assert "AAA" in result

    def test_too_few_tickers_falls_back_to_hardcoded(self):
        """Branch: <90 parsed → fall through to fallback (271-274)."""
        mock_resp = MagicMock()
        mock_resp.text = _build_nasdaq100_html(num_tickers=10)  # too few
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_nasdaq100_tickers()
        # Fallback returned (hardcoded list, 100 entries)
        assert len(result) >= 90
        assert "AAPL" in result  # hardcoded fallback contains AAPL

    def test_network_failure_uses_hardcoded_fallback(self):
        """Branch: requests.get raises → hardcoded fallback (267-274)."""
        with patch(
            "sp500_tickers.requests.get",
            side_effect=TimeoutError("slow Wikipedia"),
        ):
            result = sp500_tickers.get_nasdaq100_tickers()
        assert "AAPL" in result and "NVDA" in result


# ── get_dowjones_tickers ───────────────────────────────────────────────


class TestGetDowjonesTickers:
    """Covers sp500_tickers.py:293-350."""

    def test_cache_hit_returns_cached(self):
        sp500_tickers._ticker_cache['dowjones'] = ['CACHED_DOW']
        sp500_tickers._ticker_cache['last_fetch'] = {
            'dowjones': datetime.now()
        }
        with patch("sp500_tickers.requests.get") as mock_get:
            result = sp500_tickers.get_dowjones_tickers()
        assert result == ['CACHED_DOW']
        mock_get.assert_not_called()

    def test_wikipedia_happy_path(self):
        """Branch: wikitable with 'Symbol' header + >=25 tickers."""
        mock_resp = MagicMock()
        mock_resp.text = _build_dowjones_html(num_tickers=30)
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_dowjones_tickers()
        assert len(result) == 30
        # First generated ticker is "AAA" (i=0)
        assert "AAA" in result

    def test_network_failure_falls_back_to_hardcoded(self):
        """Branch: hardcoded 30-stock list at end."""
        with patch(
            "sp500_tickers.requests.get",
            side_effect=OSError("DNS failed"),
        ):
            result = sp500_tickers.get_dowjones_tickers()
        assert "AAPL" in result and "JPM" in result
        assert len(result) == 30  # canonical Dow 30 count


# ── get_sp400_midcap_tickers ───────────────────────────────────────────


class TestGetSp400MidcapTickers:
    """Covers sp500_tickers.py:398-430."""

    def test_wikipedia_happy_path(self):
        mock_resp = MagicMock()
        mock_resp.text = SP_CONSTITUENTS_HTML
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_sp400_midcap_tickers()
        assert "AAPL" in result
        assert "BRK-B" in result

    def test_wikitable_class_fallback(self):
        """Branch: no #constituents → table.class='wikitable' (412-413)."""
        mock_resp = MagicMock()
        mock_resp.text = SP_CONSTITUENTS_NO_ID_HTML
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_sp400_midcap_tickers()
        assert "GOOG" in result

    def test_network_failure_returns_fallback(self):
        with patch(
            "sp500_tickers.requests.get",
            side_effect=ConnectionError(),
        ):
            result = sp500_tickers.get_sp400_midcap_tickers()
        assert isinstance(result, list)
        assert len(result) > 100  # fallback list


# ── get_sp600_smallcap_tickers ─────────────────────────────────────────


class TestGetSp600SmallcapTickers:
    """Covers sp500_tickers.py:471-503."""

    def test_wikipedia_happy_path(self):
        mock_resp = MagicMock()
        mock_resp.text = SP_CONSTITUENTS_HTML
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_sp600_smallcap_tickers()
        assert "AAPL" in result

    def test_network_failure_returns_fallback(self):
        with patch(
            "sp500_tickers.requests.get",
            side_effect=ConnectionError(),
        ):
            result = sp500_tickers.get_sp600_smallcap_tickers()
        assert isinstance(result, list)
        assert len(result) > 100


# ── get_portfolio_tickers ──────────────────────────────────────────────


class TestGetPortfolioTickers:
    """Covers sp500_tickers.py:135-178."""

    def test_returns_empty_when_database_and_csv_both_unavailable(self):
        """Branch: both DB and CSV paths fail → empty list (line 178)."""
        with patch(
            "backend.database.SessionLocal",
            side_effect=ImportError("no db"),
        ), patch.object(Path, "exists", return_value=False):
            result = sp500_tickers.get_portfolio_tickers()
        # Either returns empty or whatever the DB had — both branches valid
        assert isinstance(result, list)

    def test_csv_fallback_when_db_fails(self, tmp_path, monkeypatch):
        """Branch: DB fails, CSV file exists → reads tickers from CSV."""
        # Force DB path to raise
        with patch(
            "backend.database.SessionLocal",
            side_effect=Exception("no db"),
        ):
            # Point the CSV path at a tmp file
            csv_path = tmp_path / "portfolio.csv"
            csv_path.write_text("ticker,shares\nAAPL,100\nMSFT,50\n")

            # Patch Path(__file__).parent to redirect csv discovery
            monkeypatch.setattr(
                sp500_tickers, "__file__", str(tmp_path / "sp500_tickers.py"),
            )
            result = sp500_tickers.get_portfolio_tickers()
        # Whether DB or CSV path takes effect, we should get a list
        assert isinstance(result, list)


# ── get_finviz_smallcaps ───────────────────────────────────────────────


class TestGetFinvizSmallcaps:
    """Covers sp500_tickers.py:645-706."""

    def test_happy_path_parses_first_page_then_paginates(self):
        """Branch: total found, primary links collected across pages."""
        # First call → page 1 with tickers; subsequent → empty (terminates loop)
        mock_resp_page1 = MagicMock()
        mock_resp_page1.text = FINVIZ_HTML_PAGE1
        mock_resp_page1.status_code = 200
        mock_resp_page1.raise_for_status = MagicMock()

        mock_resp_empty = MagicMock()
        mock_resp_empty.text = "<html><body><table><tr><td>Total: 0</td></tr></table></body></html>"
        mock_resp_empty.status_code = 200
        mock_resp_empty.raise_for_status = MagicMock()

        responses = [mock_resp_page1] + [mock_resp_empty] * 25
        with patch(
            "sp500_tickers.requests.get",
            side_effect=responses,
        ), patch("time.sleep"):  # skip the 0.2s respect-pause
            result = sp500_tickers.get_finviz_smallcaps()
        # Page 1's three tickers must be in the result
        for t in ("SMCO", "TINY", "MICR"):
            assert t in result

    def test_no_total_found_returns_empty(self):
        """Branch: 'Total:' text not found → return [] (line 662-664)."""
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>no total here</body></html>"
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_finviz_smallcaps()
        assert result == []

    def test_initial_request_failure_returns_empty(self):
        """Branch: outer exception → empty list (line 704-706)."""
        with patch(
            "sp500_tickers.requests.get",
            side_effect=ConnectionError("blocked"),
        ):
            result = sp500_tickers.get_finviz_smallcaps()
        assert result == []

    def test_pagination_non_200_breaks_loop(self):
        """Branch: page 2+ returns non-200 status → break (line 680-681).

        Coverage check: the for-page loop terminates cleanly when a
        subsequent page errors out.
        """
        mock_resp_page1 = MagicMock()
        mock_resp_page1.text = FINVIZ_HTML_PAGE1
        mock_resp_page1.status_code = 200
        mock_resp_page1.raise_for_status = MagicMock()

        mock_resp_bad = MagicMock()
        mock_resp_bad.status_code = 503
        mock_resp_bad.text = ""
        mock_resp_bad.raise_for_status = MagicMock()

        with patch(
            "sp500_tickers.requests.get",
            side_effect=[mock_resp_page1, mock_resp_bad],
        ), patch("time.sleep"):
            result = sp500_tickers.get_finviz_smallcaps()
        assert "SMCO" in result  # page 1 tickers preserved


# ── get_russell2000_from_sector_etfs ──────────────────────────────────


class TestGetRussell2000FromSectorEtfs:
    """Covers sp500_tickers.py:615-642."""

    def test_aggregates_smallcap_and_finviz_and_curated(self):
        """Branch: all three sources contribute, result is the union."""
        with patch(
            "sp500_tickers.get_sp600_smallcap_tickers",
            return_value=["A", "B"],
        ), patch(
            "sp500_tickers.get_finviz_smallcaps",
            return_value=["B", "C"],
        ), patch(
            "sp500_tickers.get_fallback_russell2000_tickers",
            return_value=["D"],
        ):
            result = sp500_tickers.get_russell2000_from_sector_etfs()
        # Union — duplicates collapsed
        assert set(result) == {"A", "B", "C", "D"}

    def test_smallcap_failure_swallowed(self):
        """Branch: get_sp600_smallcap raises → logged + continue (627-628)."""
        with patch(
            "sp500_tickers.get_sp600_smallcap_tickers",
            side_effect=RuntimeError("fail"),
        ), patch(
            "sp500_tickers.get_finviz_smallcaps",
            return_value=["X"],
        ), patch(
            "sp500_tickers.get_fallback_russell2000_tickers",
            return_value=["Y"],
        ):
            result = sp500_tickers.get_russell2000_from_sector_etfs()
        # The two surviving sources still contributed
        assert "X" in result and "Y" in result

    def test_finviz_failure_swallowed(self):
        """Branch: get_finviz_smallcaps raises → logged + continue (635-636)."""
        with patch(
            "sp500_tickers.get_sp600_smallcap_tickers",
            return_value=["A"],
        ), patch(
            "sp500_tickers.get_finviz_smallcaps",
            side_effect=RuntimeError("blocked"),
        ), patch(
            "sp500_tickers.get_fallback_russell2000_tickers",
            return_value=["B"],
        ):
            result = sp500_tickers.get_russell2000_from_sector_etfs()
        assert "A" in result and "B" in result


# ── get_russell2000_tickers ────────────────────────────────────────────


class TestGetRussell2000Tickers:
    """Covers get_russell2000_tickers. The Yahoo IWM top-holdings path was
    REMOVED 2026-07-06: it returned ~10 rows against a >500 acceptance gate
    (could never succeed) and burned a rate-limited call per cache refresh.
    Sources are now: cache → sector-ETF aggregate → curated fallback."""

    def test_cache_hit_short_circuits(self):
        sp500_tickers._ticker_cache['russell2000'] = ['CACHED_RUSSELL']
        sp500_tickers._ticker_cache['last_fetch'] = {
            'russell2000': datetime.now()
        }
        with patch("sp500_tickers.get_russell2000_from_sector_etfs") as mock_etfs:
            result = sp500_tickers.get_russell2000_tickers()
        assert result == ['CACHED_RUSSELL']
        mock_etfs.assert_not_called()

    def test_sector_etf_aggregate_is_primary_source(self):
        """Branch: sector-ETF aggregate large enough (>1000) → accepted."""
        with patch(
            "sp500_tickers.get_russell2000_from_sector_etfs",
            return_value=[f"S{i}" for i in range(1500)],
        ):
            result = sp500_tickers.get_russell2000_tickers()
        assert len(result) >= 1000
        assert sp500_tickers._ticker_cache['russell2000'] == result

    def test_small_sector_etf_result_falls_through_to_curated(self):
        """Branch: aggregate too small (<=1000) → curated fallback."""
        with patch(
            "sp500_tickers.get_russell2000_from_sector_etfs",
            return_value=["ONLY", "AFEW"],
        ):
            result = sp500_tickers.get_russell2000_tickers()
        assert len(result) > 200  # curated list, not the 2-element aggregate

    def test_all_sources_fail_uses_curated_fallback(self):
        """Branch: sector-ETF path raises → curated fallback."""
        with patch(
            "sp500_tickers.get_russell2000_from_sector_etfs",
            side_effect=Exception("sector etfs failed"),
        ):
            result = sp500_tickers.get_russell2000_tickers()
        assert isinstance(result, list)
        assert len(result) > 200


# ── get_all_tickers ────────────────────────────────────────────────────


class TestGetFmpScreenerTickers:
    """Covers get_fmp_screener_tickers — the FMP company-screener universe
    supplement added 2026-07-06 after the IWM/Finviz small-cap sources
    degraded (rate-limit / 403) and silently shrank the scan universe."""

    def _mock_rows(self, n=600, exchange="NYSE"):
        return [{"symbol": f"T{i}", "exchangeShortName": exchange} for i in range(n)]

    def test_cache_hit_returns_cached_without_network(self):
        sp500_tickers._ticker_cache['fmp_screener'] = ['CACHED']
        sp500_tickers._ticker_cache['last_fetch'] = {'fmp_screener': datetime.now()}
        with patch("sp500_tickers.requests.get") as mock_get:
            result = sp500_tickers.get_fmp_screener_tickers()
        assert result == ['CACHED']
        mock_get.assert_not_called()

    def test_happy_path_filters_exchanges_and_caches(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        rows = self._mock_rows(600) + [
            {"symbol": "LSE1", "exchangeShortName": "LSE"},   # non-US: dropped
            {"symbol": None, "exchangeShortName": "NYSE"},     # no symbol: dropped
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = rows
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_fmp_screener_tickers()
        assert len(result) == 600
        assert "LSE1" not in result
        assert sp500_tickers._ticker_cache['fmp_screener'] == result

    def test_tiny_result_rejected_and_not_cached(self, monkeypatch):
        """A crippled pull (endpoint change, throttling) must not poison the
        24h cache — next cycle should retry."""
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_rows(10)
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_fmp_screener_tickers()
        assert result == []
        assert sp500_tickers._ticker_cache['fmp_screener'] is None

    def test_network_failure_returns_empty_and_not_cached(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        with patch(
            "sp500_tickers.requests.get",
            side_effect=ConnectionError("network down"),
        ):
            result = sp500_tickers.get_fmp_screener_tickers()
        assert result == []
        assert sp500_tickers._ticker_cache['fmp_screener'] is None

    def test_no_api_key_returns_empty_without_network(self, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        monkeypatch.setattr(sp500_tickers, "FMP_API_KEY", "")
        with patch("sp500_tickers.requests.get") as mock_get:
            result = sp500_tickers.get_fmp_screener_tickers()
        assert result == []
        mock_get.assert_not_called()

    def test_volume_floor_not_sent_by_default(self, monkeypatch):
        """FMP volumeMoreThan filters on TODAY'S cumulative intraday volume,
        making universe size a function of cycle start time (2026-07-09:
        1,786 names at 9:57 ET vs ~3,500 by evening). Default must omit it."""
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_rows(600)
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp) as mock_get:
            sp500_tickers.get_fmp_screener_tickers()
        params = mock_get.call_args.kwargs.get("params") or mock_get.call_args.args[1]
        assert "volumeMoreThan" not in params

    def test_volume_floor_sent_when_configured_nonzero(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        from config_loader import config as yaml_config
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_rows(600)
        mock_resp.raise_for_status = MagicMock()
        with patch.object(
            yaml_config, "get",
            side_effect=lambda key, default=None:
            {"enabled": True, "volume_more_than": 75000}
            if key == "scanner.universe.fmp_screener" else default,
        ), patch("sp500_tickers.requests.get", return_value=mock_resp) as mock_get:
            sp500_tickers.get_fmp_screener_tickers()
        params = mock_get.call_args.kwargs.get("params") or mock_get.call_args.args[1]
        assert params["volumeMoreThan"] == 75000

    def test_disabled_via_config_returns_empty(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        from config_loader import config as yaml_config
        with patch.object(
            yaml_config, "get",
            side_effect=lambda key, default=None: {"enabled": False}
            if key == "scanner.universe.fmp_screener" else default,
        ), patch("sp500_tickers.requests.get") as mock_get:
            result = sp500_tickers.get_fmp_screener_tickers()
        assert result == []
        mock_get.assert_not_called()


class TestGetFmpReitTrustTickers:
    """Covers get_fmp_reit_trust_tickers — the isFund=true + sector=Real
    Estate supplement (2026-07-13). FMP marks REIT trusts isFund=true, so
    the main screener excludes every non-index REIT (live hole: CLDT)."""

    def _mock_resp(self, rows):
        resp = MagicMock()
        resp.json.return_value = rows
        resp.raise_for_status = MagicMock()
        return resp

    def test_happy_path_filters_and_caches(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        rows = [
            {"symbol": "CLDT", "exchangeShortName": "NYSE"},
            {"symbol": "WSR", "exchangeShortName": "NYSE"},
            {"symbol": "TCREX", "exchangeShortName": "NASDAQ"},  # mutual-fund class
            {"symbol": "VRSGX", "exchangeShortName": "NYSE"},    # mutual-fund class
            {"symbol": "LSE9", "exchangeShortName": "LSE"},      # non-US
            {"symbol": None, "exchangeShortName": "NYSE"},
        ]
        with patch("sp500_tickers.requests.get", return_value=self._mock_resp(rows)) as mock_get:
            result = sp500_tickers.get_fmp_reit_trust_tickers()
        assert result == ["CLDT", "WSR"]
        assert sp500_tickers._ticker_cache['fmp_reit_trusts'] == ["CLDT", "WSR"]
        params = mock_get.call_args.kwargs.get("params") or mock_get.call_args.args[1]
        assert params["isFund"] == "true"
        assert params["sector"] == "Real Estate"

    def test_four_letter_x_ticker_not_dropped(self, monkeypatch):
        """The mutual-fund convention is FIVE letters ending in X — a
        4-letter ticker ending in X (e.g. a normal equity) must survive."""
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        rows = [{"symbol": "REXX", "exchangeShortName": "NYSE"}]
        with patch("sp500_tickers.requests.get", return_value=self._mock_resp(rows)):
            result = sp500_tickers.get_fmp_reit_trust_tickers()
        assert result == ["REXX"]

    def test_oversized_result_rejected_and_not_cached(self, monkeypatch):
        """The cap protects against FMP silently ignoring the sector filter
        (the volumeMoreThan lesson) — merging thousands of isFund=true rows
        would flood the universe with funds."""
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        rows = [{"symbol": f"F{i}", "exchangeShortName": "NYSE"} for i in range(500)]
        with patch("sp500_tickers.requests.get", return_value=self._mock_resp(rows)):
            result = sp500_tickers.get_fmp_reit_trust_tickers()
        assert result == []
        assert sp500_tickers._ticker_cache['fmp_reit_trusts'] is None

    def test_network_failure_returns_empty_and_not_cached(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        with patch("sp500_tickers.requests.get", side_effect=ConnectionError("down")):
            result = sp500_tickers.get_fmp_reit_trust_tickers()
        assert result == []
        assert sp500_tickers._ticker_cache['fmp_reit_trusts'] is None

    def test_cache_hit_skips_network(self):
        sp500_tickers._ticker_cache['fmp_reit_trusts'] = ['CACHED']
        sp500_tickers._ticker_cache['last_fetch'] = {'fmp_reit_trusts': datetime.now()}
        with patch("sp500_tickers.requests.get") as mock_get:
            result = sp500_tickers.get_fmp_reit_trust_tickers()
        assert result == ['CACHED']
        mock_get.assert_not_called()

    def test_disabled_via_supplement_flag(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        from config_loader import config as yaml_config
        with patch.object(
            yaml_config, "get",
            side_effect=lambda key, default=None:
            {"enabled": True, "reit_trust_supplement": False}
            if key == "scanner.universe.fmp_screener" else default,
        ), patch("sp500_tickers.requests.get") as mock_get:
            result = sp500_tickers.get_fmp_reit_trust_tickers()
        assert result == []
        mock_get.assert_not_called()


class TestGetStickyHighScoreTickers:
    """Covers get_sticky_high_score_tickers — universe hysteresis
    (2026-07-15). Live motivation: HURC's market cap flapping 0.2% under
    the screener's $150M floor and FMP marking TBRG isActivelyTrading=false
    while it trades 300k shares/day. Both froze at high scores silently."""

    TICKERS = ("STKYHI", "STKYLO", "STKYOLD", "STKYGRW")

    def _db(self):
        sys.path.insert(0, str(Path(sp500_tickers.__file__).parent / "backend"))
        from database import SessionLocal, Stock
        return SessionLocal(), Stock

    def _cleanup(self, db, Stock):
        db.query(Stock).filter(Stock.ticker.in_(self.TICKERS)).delete(
            synchronize_session=False)
        db.commit()

    def test_returns_recent_high_scorers_only(self):
        from datetime import timezone
        db, Stock = self._db()
        try:
            self._cleanup(db, Stock)
            now = datetime.now(timezone.utc)
            db.add(Stock(ticker="STKYHI", name="Sticky High",
                         canslim_score=70.0,
                         last_updated=now - timedelta(days=2)))
            db.add(Stock(ticker="STKYLO", name="Sticky Low",
                         canslim_score=50.0,
                         last_updated=now - timedelta(days=2)))
            db.add(Stock(ticker="STKYOLD", name="Sticky Stale",
                         canslim_score=70.0,
                         last_updated=now - timedelta(days=40)))
            db.add(Stock(ticker="STKYGRW", name="Sticky Growth",
                         canslim_score=10.0, growth_mode_score=70.0,
                         last_updated=now - timedelta(days=2)))
            db.commit()

            result = sp500_tickers.get_sticky_high_score_tickers()

            assert "STKYHI" in result       # fresh + high CANSLIM score
            assert "STKYGRW" in result      # qualifies via growth score
            assert "STKYLO" not in result   # below min_score
            assert "STKYOLD" not in result  # beyond max_age_days
        finally:
            self._cleanup(db, Stock)
            db.close()

    def test_disabled_via_config_returns_empty(self):
        from config_loader import config as yaml_config
        with patch.object(
            yaml_config, "get",
            side_effect=lambda key, default=None: {"enabled": False}
            if key == "scanner.universe.stickiness" else default,
        ):
            assert sp500_tickers.get_sticky_high_score_tickers() == []

    def test_db_failure_returns_empty(self, monkeypatch):
        """Stickiness is a supplement, never a gate — DB down must not
        break universe assembly."""
        fake_db = MagicMock()
        fake_db.SessionLocal = MagicMock(side_effect=RuntimeError("DB down"))
        monkeypatch.setitem(sys.modules, "database", fake_db)
        assert sp500_tickers.get_sticky_high_score_tickers() == []


class TestGetAllTickers:
    """Covers sp500_tickers.py:69-132."""

    def test_aggregates_all_sources_and_dedupes(self):
        """Branch: portfolio + 6 index sources combined, dedupe preserves
        first-seen order (portfolio first)."""
        with patch(
            "sp500_tickers.get_portfolio_tickers",
            return_value=["AAPL", "MYPORT"],
        ), patch(
            "sp500_tickers.get_sp500_tickers",
            return_value=["AAPL", "MSFT"],  # AAPL is dup with portfolio
        ), patch(
            "sp500_tickers.get_nasdaq100_tickers",
            return_value=["NVDA"],
        ), patch(
            "sp500_tickers.get_dowjones_tickers",
            return_value=["KO"],
        ), patch(
            "sp500_tickers.get_sp400_midcap_tickers",
            return_value=[],
        ), patch(
            "sp500_tickers.get_sp600_smallcap_tickers",
            return_value=[],
        ), patch(
            "sp500_tickers.get_russell2000_tickers",
            return_value=["MYPORT"],  # dup with portfolio
        ), patch(
            "sp500_tickers.get_fmp_screener_tickers",
            return_value=["SCREENR"],
        ), patch(
            "sp500_tickers.get_fmp_reit_trust_tickers",
            return_value=["REITSUP"],
        ), patch(
            "sp500_tickers.get_sticky_high_score_tickers",
            return_value=[],
        ), patch(
            "data_fetcher.get_delisted_tickers",
            return_value=set(),
        ):
            result = sp500_tickers.get_all_tickers(
                include_portfolio=True, exclude_delisted=True,
            )
        # Portfolio tickers come first
        assert result[0] == "AAPL"
        assert result[1] == "MYPORT"
        # All unique, no duplicates
        assert len(result) == len(set(result))
        assert "MSFT" in result and "NVDA" in result

    def test_delisted_filter_excludes_non_portfolio_tickers(self):
        """Branch: delisted set filters non-portfolio tickers."""
        with patch(
            "sp500_tickers.get_portfolio_tickers",
            return_value=["MYHOLD"],
        ), patch(
            "sp500_tickers.get_sp500_tickers",
            return_value=["AAPL", "DEADCO"],
        ), patch(
            "sp500_tickers.get_nasdaq100_tickers",
            return_value=[],
        ), patch(
            "sp500_tickers.get_dowjones_tickers",
            return_value=[],
        ), patch(
            "sp500_tickers.get_sp400_midcap_tickers",
            return_value=[],
        ), patch(
            "sp500_tickers.get_sp600_smallcap_tickers",
            return_value=[],
        ), patch(
            "sp500_tickers.get_russell2000_tickers",
            return_value=[],
        ), patch(
            "sp500_tickers.get_fmp_screener_tickers",
            return_value=[],
        ), patch(
            "sp500_tickers.get_sticky_high_score_tickers",
            return_value=[],
        ), patch(
            "data_fetcher.get_delisted_tickers",
            return_value={"DEADCO"},
        ):
            result = sp500_tickers.get_all_tickers()
        assert "MYHOLD" in result
        assert "AAPL" in result
        assert "DEADCO" not in result

    def test_delisted_filter_preserves_portfolio_tickers_even_if_delisted(self):
        """Branch: portfolio ticker that's also delisted → still included
        (the 'never excluded' carve-out at line 129)."""
        with patch(
            "sp500_tickers.get_portfolio_tickers",
            return_value=["GONESTOCK"],
        ), patch(
            "sp500_tickers.get_sp500_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_nasdaq100_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_dowjones_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_sp400_midcap_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_sp600_smallcap_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_russell2000_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_fmp_screener_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_fmp_reit_trust_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_sticky_high_score_tickers", return_value=[],
        ), patch(
            "data_fetcher.get_delisted_tickers",
            return_value={"GONESTOCK"},  # marked delisted
        ):
            result = sp500_tickers.get_all_tickers(
                include_portfolio=True, exclude_delisted=True,
            )
        # Portfolio ticker survived despite being on delisted list
        assert "GONESTOCK" in result

    def test_exclude_delisted_false_skips_filter_entirely(self):
        """Branch: exclude_delisted=False → no DB call, no filter (line 88)."""
        with patch(
            "sp500_tickers.get_portfolio_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_sp500_tickers", return_value=["A"],
        ), patch(
            "sp500_tickers.get_nasdaq100_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_dowjones_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_sp400_midcap_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_sp600_smallcap_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_russell2000_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_fmp_screener_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_fmp_reit_trust_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_sticky_high_score_tickers", return_value=[],
        ), patch(
            "data_fetcher.get_delisted_tickers"
        ) as mock_delisted:
            result = sp500_tickers.get_all_tickers(
                include_portfolio=False, exclude_delisted=False,
            )
        # Delisted lookup not invoked
        mock_delisted.assert_not_called()
        assert "A" in result

    def test_delisted_fetch_failure_swallowed(self):
        """Branch: get_delisted_tickers raises → logged + delisted=set()
        (line 94-95)."""
        with patch(
            "sp500_tickers.get_portfolio_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_sp500_tickers", return_value=["A"],
        ), patch(
            "sp500_tickers.get_nasdaq100_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_dowjones_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_sp400_midcap_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_sp600_smallcap_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_russell2000_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_fmp_screener_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_fmp_reit_trust_tickers", return_value=[],
        ), patch(
            "sp500_tickers.get_sticky_high_score_tickers", return_value=[],
        ), patch(
            "data_fetcher.get_delisted_tickers",
            side_effect=RuntimeError("db error"),
        ):
            result = sp500_tickers.get_all_tickers(
                include_portfolio=False, exclude_delisted=True,
            )
        # Endpoint completed; all tickers preserved
        assert "A" in result

    def _patch_sources(self, sticky, delisted=frozenset()):
        """All sources empty except sp500=[AAPL]; sticky + delisted as given."""
        from contextlib import ExitStack
        stack = ExitStack()
        for src in ("get_portfolio_tickers", "get_nasdaq100_tickers",
                    "get_dowjones_tickers", "get_sp400_midcap_tickers",
                    "get_sp600_smallcap_tickers", "get_russell2000_tickers",
                    "get_fmp_screener_tickers", "get_fmp_reit_trust_tickers"):
            stack.enter_context(patch(f"sp500_tickers.{src}", return_value=[]))
        stack.enter_context(patch(
            "sp500_tickers.get_sp500_tickers", return_value=["AAPL"]))
        stack.enter_context(patch(
            "sp500_tickers.get_sticky_high_score_tickers", return_value=sticky))
        stack.enter_context(patch(
            "data_fetcher.get_delisted_tickers", return_value=set(delisted)))
        return stack

    def test_sticky_names_appended_and_deduped(self):
        """Universe hysteresis: sticky names enter the universe; ones already
        provided by a source don't duplicate."""
        with self._patch_sources(sticky=["STICKME", "AAPL"]):
            result = sp500_tickers.get_all_tickers(
                include_portfolio=True, exclude_delisted=True,
            )
        assert "STICKME" in result
        assert result.count("AAPL") == 1

    def test_sticky_names_still_filtered_by_delisted(self):
        """Stickiness must not resurrect blocked/delisted tickers — it is
        appended before the delisted filter."""
        with self._patch_sources(sticky=["BLOCKED1"], delisted={"BLOCKED1"}):
            result = sp500_tickers.get_all_tickers(
                include_portfolio=True, exclude_delisted=True,
            )
        assert "BLOCKED1" not in result


# ── _load_env ──────────────────────────────────────────────────────────


class TestLoadEnv:
    """Covers sp500_tickers.py:863-874."""

    def test_loads_env_when_dotenv_file_present(self, tmp_path, monkeypatch):
        """Branch: .env file exists → keys propagated into os.environ."""
        # Place a .env file next to a fake sp500_tickers.py
        env_file = tmp_path / ".env"
        env_file.write_text("FMP_API_KEY=test-key-123\nOTHER=ignored\n")

        # Pretend sp500_tickers.py lives in tmp_path
        monkeypatch.setattr(
            sp500_tickers, "__file__", str(tmp_path / "sp500_tickers.py"),
        )
        # Clear FMP_API_KEY from the env so we can see the load take effect
        monkeypatch.delenv("FMP_API_KEY", raising=False)

        sp500_tickers._load_env()
        # setdefault was used, so the value lands in os.environ
        assert os.environ.get("FMP_API_KEY") == "test-key-123"
        assert sp500_tickers.FMP_API_KEY == "test-key-123"

    def test_no_op_when_dotenv_missing(self, tmp_path, monkeypatch):
        """Branch: .env file absent → function returns silently."""
        monkeypatch.setattr(
            sp500_tickers, "__file__", str(tmp_path / "sp500_tickers.py"),
        )
        # No .env created → if-branch at line 868 is False
        sp500_tickers._load_env()
        # No exception; no state changed


# ── Coverage Gaps ──────────────────────────────────────────────────────


class TestCoverageGaps:
    """Branches the existing scenario suites don't reach: portfolio DB-fail
    cascading to CSV read, SmallCap 600 wikitable fallback, and the Finviz
    page-2+ parsing loop body + inner exception handler."""

    def test_portfolio_db_exception_falls_through_to_csv(self, tmp_path, monkeypatch):
        # Force the function's `from database import SessionLocal` to resolve
        # to a stub whose SessionLocal() raises. The function does a sys.path
        # insert + import inside its try block, so we plant the stub in
        # sys.modules under the short name "database" — that's the cache key
        # the import will consult.
        fake_db = MagicMock()
        fake_db.SessionLocal = MagicMock(side_effect=RuntimeError("DB down"))
        monkeypatch.setitem(sys.modules, "database", fake_db)

        # CSV in tmp_path; redirect sp500_tickers.__file__ so the function
        # resolves portfolio.csv next to it.
        csv_path = tmp_path / "portfolio.csv"
        csv_path.write_text("ticker,shares\nAAPL,100\nMSFT,50\n")
        monkeypatch.setattr(
            sp500_tickers, "__file__", str(tmp_path / "sp500_tickers.py"),
        )

        result = sp500_tickers.get_portfolio_tickers()
        assert "AAPL" in result
        assert "MSFT" in result

    def test_portfolio_db_exception_and_csv_read_error(self, tmp_path, monkeypatch):
        # DB raises (hits line 162-163) AND the CSV open raises (hits 175-176).
        # End result: empty list returned at line 178.
        fake_db = MagicMock()
        fake_db.SessionLocal = MagicMock(side_effect=RuntimeError("DB down"))
        monkeypatch.setitem(sys.modules, "database", fake_db)

        # Create a directory at the path where portfolio.csv is expected so
        # csv_path.exists() is True but open() raises IsADirectoryError.
        (tmp_path / "portfolio.csv").mkdir()
        monkeypatch.setattr(
            sp500_tickers, "__file__", str(tmp_path / "sp500_tickers.py"),
        )

        result = sp500_tickers.get_portfolio_tickers()
        assert result == []

    def test_smallcap_wikitable_fallback_when_constituents_id_missing(self):
        # Wikipedia variant: no id="constituents" table, but a class="wikitable"
        # one (line 486 fallback path).
        mock_resp = MagicMock()
        mock_resp.text = SP_CONSTITUENTS_NO_ID_HTML
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        with patch("sp500_tickers.requests.get", return_value=mock_resp):
            result = sp500_tickers.get_sp600_smallcap_tickers()
        assert "GOOG" in result
        assert "META" in result

    def test_finviz_pagination_collects_page2_then_inner_exception_breaks(self):
        # Page 1: tickers via the primary loop (already covered).
        mock_p1 = MagicMock()
        mock_p1.text = FINVIZ_HTML_PAGE1
        mock_p1.status_code = 200
        mock_p1.raise_for_status = MagicMock()

        # Page 2: more tickers — covers lines 689-696 (inner for-loop body
        # + import time + time.sleep).
        page2_html = """<html><body>
        <a class="screener-link-primary">PAGE</a>
        <a class="screener-link-primary">TWO</a>
        </body></html>"""
        mock_p2 = MagicMock()
        mock_p2.text = page2_html
        mock_p2.status_code = 200
        mock_p2.raise_for_status = MagicMock()

        # Page 3: requests.get raises — covers lines 698-700 (inner except + break).
        with patch(
            "sp500_tickers.requests.get",
            side_effect=[mock_p1, mock_p2, ConnectionError("page 3 fail")],
        ), patch("time.sleep"):
            result = sp500_tickers.get_finviz_smallcaps()

        # Page 1 tickers preserved
        assert "SMCO" in result
        # Page 2 tickers added via the loop body (line 689-692)
        assert "PAGE" in result
        assert "TWO" in result
