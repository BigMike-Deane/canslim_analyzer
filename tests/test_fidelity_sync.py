"""Tests for Fidelity CSV parsing and reconciliation."""
import pytest
from backend.fidelity_sync import (
    parse_positions_csv,
    parse_activity_csv,
    reconcile_portfolios,
    _clean_dollar,
    _clean_percent,
    _clean_float,
    _is_option_symbol,
    _aggregate_fills,
)


# ============== Value Cleaning Tests ==============

class TestCleanDollar:
    def test_positive_with_sign(self):
        assert _clean_dollar('+$1,234.56') == 1234.56

    def test_negative(self):
        assert _clean_dollar('-$73.00') == -73.0

    def test_plain(self):
        assert _clean_dollar('$6534.41') == 6534.41

    def test_dash_returns_none(self):
        assert _clean_dollar('--') is None

    def test_empty_returns_none(self):
        assert _clean_dollar('') is None
        assert _clean_dollar(None) is None


class TestCleanPercent:
    def test_positive(self):
        assert _clean_percent('+2.36%') == 2.36

    def test_negative(self):
        assert _clean_percent('-77.58%') == -77.58

    def test_dash_returns_none(self):
        assert _clean_percent('--') is None


class TestCleanFloat:
    def test_integer(self):
        assert _clean_float('1000') == 1000.0

    def test_with_commas(self):
        assert _clean_float('1,234.5') == 1234.5

    def test_with_plus(self):
        assert _clean_float('+0.045') == 0.045


class TestIsOptionSymbol:
    def test_fidelity_option_format(self):
        assert _is_option_symbol('-ONDS260327C13') is True

    def test_normal_stock(self):
        assert _is_option_symbol('AAPL') is False
        assert _is_option_symbol('ARM') is False

    def test_empty(self):
        assert _is_option_symbol('') is False
        assert _is_option_symbol(None) is False


# ============== Positions CSV Parsing Tests ==============

SAMPLE_POSITIONS_CSV = """Account Number,Account Name,Symbol,Description,Quantity,Last Price,Last Price Change,Current Value,Today's Gain/Loss Dollar,Today's Gain/Loss Percent,Total Gain/Loss Dollar,Total Gain/Loss Percent,Percent Of Account,Cost Basis Total,Average Cost Basis,Type
Z27804829,Individual,SPAXX**,HELD IN MONEY MARKET,,,,$6534.41,,,,,36.69%,,,Cash,
Z27804829,Individual,LCTX,LINEAGE CELL THERAPEUTICS INC COM,1000,$1.945,+$0.045,$1945.00,+$45.00,+2.36%,+$259.75,+15.41%,10.92%,$1685.25,$1.69,Margin,
Z27804829,Individual,HUMA,HUMACYTE INC COM,430,$1.2052,+$0.0452,$518.23,+$19.43,+3.89%,-$1792.95,-77.58%,2.91%,$2311.18,$5.37,Margin,
Z27804829,Individual,-ONDS260327C13,ONDS MAR 27 2026 $13 CALL,-1,$0.73,+$0.12,-$73.00,-$12.00,-19.68%,+$1.33,+1.78%,-0.41%,$74.33,$0.74,Margin,
41149,401K,FNSBX**,FIDELITY SUSTAINABLE BOND IDX,123.45,$10.00,,$1234.50,,,,,,,,Cash,

"Date downloaded Feb-26-2026 11:27 a.m ET"
"""


class TestParsePositionsCsv:
    def test_parses_positions(self):
        result = parse_positions_csv(SAMPLE_POSITIONS_CSV)
        assert result["account"] == "Z27804829"
        assert len(result["positions"]) == 2  # LCTX and HUMA (option skipped, 401K skipped)

    def test_filters_to_target_account(self):
        result = parse_positions_csv(SAMPLE_POSITIONS_CSV)
        symbols = [p["symbol"] for p in result["positions"]]
        assert "LCTX" in symbols
        assert "HUMA" in symbols
        assert "FNSBX" not in symbols  # 401K account

    def test_skips_options(self):
        result = parse_positions_csv(SAMPLE_POSITIONS_CSV)
        symbols = [p["symbol"] for p in result["positions"]]
        assert "-ONDS260327C13" not in symbols

    def test_extracts_cash_balance(self):
        result = parse_positions_csv(SAMPLE_POSITIONS_CSV)
        assert result["cash_balance"] == 6534.41

    def test_parses_dollar_values(self):
        result = parse_positions_csv(SAMPLE_POSITIONS_CSV)
        lctx = next(p for p in result["positions"] if p["symbol"] == "LCTX")
        assert lctx["quantity"] == 1000
        assert lctx["last_price"] == 1.945
        assert lctx["current_value"] == 1945.00
        assert lctx["average_cost_basis"] == 1.69
        assert lctx["total_gain_loss"] == 259.75

    def test_parses_percent_values(self):
        result = parse_positions_csv(SAMPLE_POSITIONS_CSV)
        huma = next(p for p in result["positions"] if p["symbol"] == "HUMA")
        assert huma["total_gain_loss_pct"] == -77.58

    def test_extracts_snapshot_date(self):
        result = parse_positions_csv(SAMPLE_POSITIONS_CSV)
        assert result["snapshot_date"] == "2026-02-26"

    def test_total_value_includes_cash(self):
        result = parse_positions_csv(SAMPLE_POSITIONS_CSV)
        # Cash (6534.41) + LCTX (1945) + HUMA (518.23)
        expected = 6534.41 + 1945.00 + 518.23
        assert abs(result["total_value"] - expected) < 0.01


# ============== Activity CSV Parsing Tests ==============

SAMPLE_ACTIVITY_CSV = """

Run Date,Account,Account Number,Action,Symbol,Description,Type,Price ($),Quantity,Commission ($),Fees ($),Accrued Interest ($),Amount ($),Settlement Date
02/25/2026,Individual,Z27804829,YOU BOUGHT ONDAS INC COMMON STOCK (ONDS) (Margin),ONDS,,Margin,10.30,50,,,,-$515.00,02/27/2026
02/25/2026,Individual,Z27804829,YOU SOLD ARM HOLDINGS PLC SPON ADS (ARM) (Margin),ARM,,Margin,130.00,-12,,,,$1560.00,02/27/2026
02/25/2026,Individual,Z27804829,YOU SOLD ARM HOLDINGS PLC SPON ADS (ARM) (Margin),ARM,,Margin,129.90,-0.082,,,,$10.65,02/27/2026
02/20/2026,Individual,Z27804829, DIVIDEND RECEIVED,NANC,,,$0.50,70,,,,$35.00,02/20/2026
02/18/2026,401K,41149,YOU BOUGHT SOMETHING (SPY),SPY,,Cash,500.00,10,,,,-$5000.00,02/20/2026
02/15/2026,Individual,Z27804829,ASSIGNED -ONDS260117C10,,,-ONDS260117C10,,,,,,,
"""


class TestParseActivityCsv:
    def test_parses_trades(self):
        result = parse_activity_csv(SAMPLE_ACTIVITY_CSV)
        assert len(result["trades"]) >= 2  # ONDS buy + ARM sell (aggregated)

    def test_filters_to_target_account(self):
        result = parse_activity_csv(SAMPLE_ACTIVITY_CSV)
        symbols = [t["symbol"] for t in result["trades"]]
        assert "SPY" not in symbols  # 401K account

    def test_classifies_buy_sell(self):
        result = parse_activity_csv(SAMPLE_ACTIVITY_CSV)
        onds = next((t for t in result["trades"] if t["symbol"] == "ONDS"), None)
        assert onds is not None
        assert onds["action"] == "BUY"

        arm = next((t for t in result["trades"] if t["symbol"] == "ARM"), None)
        assert arm is not None
        assert arm["action"] == "SELL"

    def test_aggregates_fills(self):
        result = parse_activity_csv(SAMPLE_ACTIVITY_CSV)
        arm = next(t for t in result["trades"] if t["symbol"] == "ARM")
        # Should aggregate -12 and -0.082 into 12.082
        assert abs(arm["quantity"] - 12.082) < 0.001

    def test_extracts_dividends(self):
        result = parse_activity_csv(SAMPLE_ACTIVITY_CSV)
        assert len(result["dividends"]) >= 1
        assert result["dividends"][0]["symbol"] == "NANC"
        assert result["dividends"][0]["amount"] == 35.00

    def test_skips_option_assignments(self):
        result = parse_activity_csv(SAMPLE_ACTIVITY_CSV)
        symbols = [t["symbol"] for t in result["trades"]]
        assert "-ONDS260117C10" not in symbols

    def test_handles_blank_lines_before_header(self):
        # The 2 blank lines at top should be handled
        result = parse_activity_csv(SAMPLE_ACTIVITY_CSV)
        assert len(result["parse_errors"]) == 0

    def test_date_parsing(self):
        result = parse_activity_csv(SAMPLE_ACTIVITY_CSV)
        onds = next(t for t in result["trades"] if t["symbol"] == "ONDS")
        assert onds["run_date"] == "2026-02-25"
        assert onds["settlement_date"] == "2026-02-27"


# ============== Fill Aggregation Tests ==============

class TestAggregateFills:
    def test_combines_same_day_same_action(self):
        fills = [
            {"run_date": "2026-02-25", "symbol": "ARM", "action": "SELL",
             "price": 130.00, "quantity": 12, "amount": -1560.00,
             "commission": 0, "fees": 0, "description": "ARM", "settlement_date": None, "raw_action": "SELL"},
            {"run_date": "2026-02-25", "symbol": "ARM", "action": "SELL",
             "price": 129.90, "quantity": 0.082, "amount": -10.65,
             "commission": 0, "fees": 0, "description": "ARM", "settlement_date": None, "raw_action": "SELL"},
        ]
        result = _aggregate_fills(fills)
        assert len(result) == 1
        assert abs(result[0]["quantity"] - 12.082) < 0.001

    def test_keeps_different_days_separate(self):
        fills = [
            {"run_date": "2026-02-25", "symbol": "ARM", "action": "BUY",
             "price": 130, "quantity": 10, "amount": -1300,
             "commission": 0, "fees": 0, "description": "", "settlement_date": None, "raw_action": "BUY"},
            {"run_date": "2026-02-26", "symbol": "ARM", "action": "BUY",
             "price": 131, "quantity": 5, "amount": -655,
             "commission": 0, "fees": 0, "description": "", "settlement_date": None, "raw_action": "BUY"},
        ]
        result = _aggregate_fills(fills)
        assert len(result) == 2


# ============== Reconciliation Tests ==============

class TestReconcilePortfolios:
    def test_matching_positions(self):
        fid = [{"symbol": "AAPL", "quantity": 100, "current_value": 15000, "total_gain_loss_pct": 10.5, "average_cost_basis": 135}]
        ai = [{"ticker": "AAPL", "shares": 100, "current_value": 15000, "gain_loss_pct": 10.5, "current_score": 78}]
        result = reconcile_portfolios(fid, ai)
        assert len(result["matches"]) == 1
        assert result["matches"][0]["symbol"] == "AAPL"
        assert result["matches"][0]["status"] == "match"

    def test_fidelity_only(self):
        fid = [{"symbol": "LCTX", "quantity": 1000, "current_value": 1945, "total_gain_loss_pct": 15, "average_cost_basis": 1.69}]
        ai = []
        result = reconcile_portfolios(fid, ai)
        assert len(result["fidelity_only"]) == 1
        assert result["fidelity_only"][0]["symbol"] == "LCTX"

    def test_ai_only(self):
        fid = []
        ai = [{"ticker": "NVDA", "shares": 50, "current_value": 5000, "gain_loss_pct": 20, "current_score": 85}]
        result = reconcile_portfolios(fid, ai)
        assert len(result["ai_only"]) == 1
        assert result["ai_only"][0]["symbol"] == "NVDA"

    def test_quantity_mismatch(self):
        fid = [{"symbol": "ARM", "quantity": 20, "current_value": 2600, "total_gain_loss_pct": 26, "average_cost_basis": 102}]
        ai = [{"ticker": "ARM", "shares": 32.082, "current_value": 4200, "gain_loss_pct": 15, "current_score": 72}]
        result = reconcile_portfolios(fid, ai)
        assert len(result["discrepancies"]) == 1
        assert result["discrepancies"][0]["symbol"] == "ARM"
        assert abs(result["discrepancies"][0]["share_diff"] - (20 - 32.082)) < 0.001

    def test_overlap_percentage(self):
        fid = [
            {"symbol": "AAPL", "quantity": 100, "current_value": 15000, "total_gain_loss_pct": 10, "average_cost_basis": 135},
            {"symbol": "MSFT", "quantity": 50, "current_value": 20000, "total_gain_loss_pct": 5, "average_cost_basis": 380},
        ]
        ai = [
            {"ticker": "AAPL", "shares": 100, "current_value": 15000, "gain_loss_pct": 10, "current_score": 78},
            {"ticker": "GOOGL", "shares": 30, "current_value": 5000, "gain_loss_pct": 12, "current_score": 82},
        ]
        result = reconcile_portfolios(fid, ai)
        # 3 unique symbols (AAPL, MSFT, GOOGL), 1 overlap (AAPL)
        assert result["summary"]["overlap_count"] == 1
        assert result["summary"]["total_unique_symbols"] == 3
        assert abs(result["summary"]["overlap_pct"] - 33.3) < 0.1

    def test_empty_portfolios(self):
        result = reconcile_portfolios([], [])
        assert result["summary"]["overlap_pct"] == 0
        assert result["summary"]["total_unique_symbols"] == 0
