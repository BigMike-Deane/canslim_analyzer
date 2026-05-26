"""Tests for the yfinance ERROR-noise log filter in backend.main.

The filter exists to drop unactionable yfinance ERROR spam during full-universe
scans (delisted/no-data restatements and self-healing Yahoo auth churn) while
keeping genuinely actionable errors (rate-limits, network failures) visible.

Regression context: the filter originally matched only the older download-path
strings ("possibly delisted", "Failed download", "Quote not found") and missed
the dominant 404 spam from the .info/quoteSummary path
("No fundamentals data found for symbol: X") — 13K+ lines per scan in prod.
"""

import logging

from backend.main import _YFinanceNoiseFilter


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="yfinance",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestYFinanceNoiseFilterDrops:
    """Records the filter should suppress (filter() returns False)."""

    def setup_method(self):
        self.filt = _YFinanceNoiseFilter()

    def test_drops_quotesummary_404_no_fundamentals(self):
        # The dominant prod spam (13,152 lines): .info/quoteSummary 404.
        msg = ('HTTP Error 404: {"quoteSummary":{"result":null,"error":'
               '{"code":"Not Found","description":"No fundamentals data found '
               'for symbol: SLCA"}}}')
        assert self.filt.filter(_record(msg)) is False

    def test_drops_invalid_crumb_401(self):
        msg = ('HTTP Error 401: {"finance":{"result":null,"error":'
               '{"code":"Unauthorized","description":"Invalid Crumb"}}}')
        assert self.filt.filter(_record(msg)) is False

    def test_drops_feature_access_401(self):
        msg = ('HTTP Error 401: {"finance":{"result":null,"error":'
               '{"code":"Unauthorized","description":"User is unable to access '
               'this feature - https://bit.ly/yahoo-finance-api-feedback"}}}')
        assert self.filt.filter(_record(msg)) is False

    def test_drops_legacy_possibly_delisted(self):
        assert self.filt.filter(_record("FOO: possibly delisted; no data found")) is False

    def test_drops_legacy_failed_download(self):
        assert self.filt.filter(_record("Failed download: 1 ticker dropped")) is False

    def test_drops_legacy_quote_not_found(self):
        assert self.filt.filter(_record("BAR: Quote not found for symbol")) is False


class TestYFinanceNoiseFilterKeeps:
    """Records the filter must NOT suppress (filter() returns True)."""

    def setup_method(self):
        self.filt = _YFinanceNoiseFilter()

    def test_keeps_curl_network_error(self):
        # Genuine, actionable network failure — must stay visible.
        msg = ("Failed to perform, curl: (16) . See "
               "https://curl.se/libcurl/c/libcurl-errors.html first for more details.")
        assert self.filt.filter(_record(msg)) is True

    def test_keeps_rate_limit_error(self):
        assert self.filt.filter(_record("HTTP Error 429: Too Many Requests")) is True

    def test_keeps_generic_500_error(self):
        assert self.filt.filter(_record("HTTP Error 500: Internal Server Error")) is True

    def test_keeps_plain_info_message(self):
        assert self.filt.filter(_record("AAPL: downloaded 250 rows")) is True

    def test_keeps_record_with_unformattable_message(self):
        # getMessage() raises (TypeError: %d wants a number) → fail open.
        rec = logging.LogRecord(
            name="yfinance", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="needs %d args", args=("not-an-int",), exc_info=None,
        )
        assert self.filt.filter(rec) is True
