"""
Fidelity CSV Parser and Portfolio Reconciliation

Parses Fidelity position and activity CSV exports, filters to
the configured account, and provides reconciliation against the
AI portfolio recommendations.
"""

import csv
import io
import re
import logging
from datetime import datetime, date, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Only process this Fidelity account
TARGET_ACCOUNT = "Z27804829"


def _clean_dollar(value: str) -> Optional[float]:
    """Parse Fidelity dollar values like '+$1,234.56' or '-$73.00' or '$6534.41'."""
    if not value or value.strip() in ('', '--', 'n/a'):
        return None
    cleaned = value.strip().replace('$', '').replace(',', '').replace('+', '')
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _clean_percent(value: str) -> Optional[float]:
    """Parse Fidelity percent values like '+2.36%' or '-77.58%'."""
    if not value or value.strip() in ('', '--', 'n/a'):
        return None
    cleaned = value.strip().replace('%', '').replace('+', '').replace(',', '')
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _clean_float(value: str) -> Optional[float]:
    """Parse a plain numeric value."""
    if not value or value.strip() in ('', '--', 'n/a'):
        return None
    cleaned = value.strip().replace(',', '').replace('+', '')
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _is_option_symbol(symbol: str) -> bool:
    """Detect option symbols like '-ONDS260327C13'."""
    if not symbol:
        return False
    symbol = symbol.strip()
    # Options typically start with '-' or contain date+strike patterns
    if symbol.startswith('-'):
        return True
    # Match patterns like AAPL260321C150
    if re.match(r'^[A-Z]+\d{6}[CP]\d+', symbol):
        return True
    return False


def parse_positions_csv(csv_content: str) -> dict:
    """
    Parse a Fidelity Positions CSV export.

    Returns:
        {
            "account": str,
            "positions": [
                {
                    "symbol": str,
                    "description": str,
                    "quantity": float,
                    "last_price": float,
                    "current_value": float,
                    "total_gain_loss": float,
                    "total_gain_loss_pct": float,
                    "cost_basis_total": float,
                    "average_cost_basis": float,
                    "percent_of_account": float,
                    "type": str,  # "Margin", "Cash"
                }
            ],
            "cash_balance": float,
            "total_value": float,
            "snapshot_date": str,
            "parse_errors": [str],
        }
    """
    positions = []
    cash_balance = 0.0
    total_value = 0.0
    parse_errors = []
    snapshot_date = None

    # Strip BOM if present (Fidelity exports UTF-8 with BOM)
    if csv_content.startswith('\ufeff'):
        csv_content = csv_content[1:]

    reader = csv.DictReader(io.StringIO(csv_content))

    for row in reader:
        # Skip rows not from target account
        account = (row.get('Account Number') or '').strip()
        if account != TARGET_ACCOUNT:
            continue

        symbol = (row.get('Symbol') or '').strip()
        if not symbol:
            continue

        # Extract date from disclaimer if present
        if not snapshot_date:
            # Will try to extract from end of file later
            pass

        # Skip options
        if _is_option_symbol(symbol):
            logger.debug(f"Skipping option: {symbol}")
            continue

        # Handle money market / cash positions
        if symbol.endswith('**') or 'MONEY MARKET' in (row.get('Description') or '').upper():
            cash_val = _clean_dollar(row.get('Current Value'))
            if cash_val is not None:
                cash_balance = cash_val
            continue

        # Parse position data
        try:
            quantity = _clean_float(row.get('Quantity'))
            last_price = _clean_dollar(row.get('Last Price'))
            current_value = _clean_dollar(row.get('Current Value'))
            total_gl = _clean_dollar(row.get("Total Gain/Loss Dollar"))
            total_gl_pct = _clean_percent(row.get("Total Gain/Loss Percent"))
            cost_basis_total = _clean_dollar(row.get('Cost Basis Total'))
            avg_cost_basis = _clean_dollar(row.get('Average Cost Basis'))
            pct_of_account = _clean_percent(row.get('Percent Of Account'))
            pos_type = (row.get('Type') or '').strip().rstrip(',')

            if quantity is None or quantity <= 0:
                parse_errors.append(f"Skipped {symbol}: invalid quantity")
                continue

            positions.append({
                "symbol": symbol,
                "description": (row.get('Description') or '').strip(),
                "quantity": quantity,
                "last_price": last_price,
                "current_value": current_value,
                "total_gain_loss": total_gl,
                "total_gain_loss_pct": total_gl_pct,
                "cost_basis_total": cost_basis_total,
                "average_cost_basis": avg_cost_basis,
                "percent_of_account": pct_of_account,
                "type": pos_type,
            })

            if current_value is not None:
                total_value += current_value

        except Exception as e:
            parse_errors.append(f"Error parsing {symbol}: {e}")

    total_value += cash_balance

    # Try to extract snapshot date from raw content
    date_match = re.search(r'Date downloaded\s+(\w+-\d+-\d+)', csv_content)
    if date_match:
        try:
            snapshot_date = datetime.strptime(date_match.group(1), '%b-%d-%Y').strftime('%Y-%m-%d')
        except ValueError:
            pass

    if not snapshot_date:
        snapshot_date = date.today().isoformat()

    return {
        "account": TARGET_ACCOUNT,
        "positions": positions,
        "cash_balance": cash_balance,
        "total_value": total_value,
        "snapshot_date": snapshot_date,
        "parse_errors": parse_errors,
    }


def parse_activity_csv(csv_content: str) -> dict:
    """
    Parse a Fidelity Account Activity/History CSV export.

    Returns:
        {
            "trades": [
                {
                    "run_date": str (YYYY-MM-DD),
                    "action": str,  # "BUY", "SELL"
                    "symbol": str,
                    "description": str,
                    "price": float,
                    "quantity": float,
                    "amount": float,
                    "commission": float,
                    "fees": float,
                    "settlement_date": str,
                    "raw_action": str,  # Original action text
                }
            ],
            "dividends": [...],
            "other": [...],
            "parse_errors": [str],
        }
    """
    trades = []
    dividends = []
    other = []
    parse_errors = []

    # Strip BOM if present (Fidelity exports UTF-8 with BOM)
    if csv_content.startswith('\ufeff'):
        csv_content = csv_content[1:]

    # Activity CSV has blank lines at top — skip them
    lines = csv_content.splitlines()
    # Find the header line (starts with "Run Date")
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Run Date'):
            header_idx = i
            break

    if header_idx is None:
        return {
            "trades": [], "dividends": [], "other": [],
            "parse_errors": ["Could not find header row in activity CSV"]
        }

    # Re-parse from the header line onward
    csv_body = '\n'.join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_body))

    for row in reader:
        # Skip rows not from target account
        account_num = (row.get('Account Number') or '').strip()
        if account_num != TARGET_ACCOUNT:
            continue

        raw_action = (row.get('Action') or '').strip()
        symbol = (row.get('Symbol') or '').strip()
        run_date_str = (row.get('Run Date') or '').strip()

        # Skip empty rows / disclaimer text
        if not raw_action or not run_date_str:
            continue

        # Skip options
        if _is_option_symbol(symbol):
            continue

        # Parse date (MM/DD/YYYY)
        try:
            run_date = datetime.strptime(run_date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
        except ValueError:
            parse_errors.append(f"Bad date: {run_date_str}")
            continue

        # Parse settlement date
        settlement_str = (row.get('Settlement Date') or '').strip()
        settlement_date = None
        if settlement_str:
            try:
                settlement_date = datetime.strptime(settlement_str, '%m/%d/%Y').strftime('%Y-%m-%d')
            except ValueError:
                pass

        price = _clean_dollar(row.get('Price ($)'))
        quantity = _clean_float(row.get('Quantity'))
        amount = _clean_dollar(row.get('Amount ($)'))
        commission = _clean_dollar(row.get('Commission ($)')) or 0
        fees = _clean_dollar(row.get('Fees ($)')) or 0

        # Classify the action
        action_upper = raw_action.upper()

        if 'DIVIDEND' in action_upper or 'REINVESTMENT' in action_upper:
            dividends.append({
                "run_date": run_date,
                "symbol": symbol,
                "description": (row.get('Description') or '').strip(),
                "amount": amount,
                "raw_action": raw_action,
            })
            continue

        if 'YOU BOUGHT' in action_upper:
            action = 'BUY'
        elif 'YOU SOLD' in action_upper:
            action = 'SELL'
        elif 'ASSIGNED' in action_upper or 'EXPIRED' in action_upper:
            # Options assignment/expiry — skip
            continue
        elif any(kw in action_upper for kw in ['JOURNALED', 'CONTRIBUTION', 'TRANSFER', 'WIRE']):
            other.append({
                "run_date": run_date,
                "symbol": symbol,
                "raw_action": raw_action,
                "amount": amount,
            })
            continue
        else:
            other.append({
                "run_date": run_date,
                "symbol": symbol,
                "raw_action": raw_action,
                "amount": amount,
            })
            continue

        # Normalize quantity for sells (Fidelity uses negative quantities for sells)
        abs_quantity = abs(quantity) if quantity else 0

        trades.append({
            "run_date": run_date,
            "action": action,
            "symbol": symbol,
            "description": (row.get('Description') or '').strip(),
            "price": price,
            "quantity": abs_quantity,
            "amount": amount,
            "commission": commission,
            "fees": fees,
            "settlement_date": settlement_date,
            "raw_action": raw_action,
        })

    # Aggregate same-day, same-symbol, same-action fills into single trades
    aggregated = _aggregate_fills(trades)

    return {
        "trades": aggregated,
        "dividends": dividends,
        "other": other,
        "parse_errors": parse_errors,
    }


def _aggregate_fills(trades: list) -> list:
    """
    Combine multiple fills for the same stock on the same day.
    E.g., ARM sold as -0.082 and -12 shares on same day → single trade of 12.082 shares.
    """
    key_map = {}  # (date, symbol, action) -> aggregated trade
    for t in trades:
        key = (t["run_date"], t["symbol"], t["action"])
        if key not in key_map:
            key_map[key] = {
                **t,
                "_fill_count": 1,
                "_total_amount": abs(t["amount"]) if t["amount"] else 0,
            }
        else:
            existing = key_map[key]
            existing["quantity"] += t["quantity"]
            existing["commission"] += t["commission"]
            existing["fees"] += t["fees"]
            if t["amount"]:
                existing["_total_amount"] += abs(t["amount"])
            existing["_fill_count"] += 1
            # Weighted average price
            if existing["quantity"] > 0 and existing["_total_amount"] > 0:
                existing["price"] = round(existing["_total_amount"] / existing["quantity"], 4)

    result = []
    for t in key_map.values():
        t.pop("_fill_count", None)
        t.pop("_total_amount", None)
        t["quantity"] = round(t["quantity"], 6)
        result.append(t)

    # Sort by date descending (newest first)
    result.sort(key=lambda x: x["run_date"], reverse=True)
    return result


def reconcile_portfolios(fidelity_positions: list, ai_positions: list, ai_trades: list = None) -> dict:
    """
    Compare Fidelity actual positions against AI portfolio positions.

    Returns:
        {
            "matches": [...],       # Same ticker in both
            "fidelity_only": [...], # In Fidelity but not AI
            "ai_only": [...],       # In AI but not Fidelity
            "discrepancies": [...], # Same ticker but different quantities
            "summary": {
                "fidelity_total": float,
                "ai_total": float,
                "overlap_pct": float,
            }
        }
    """
    fid_map = {p["symbol"]: p for p in fidelity_positions}
    ai_map = {p["ticker"]: p for p in ai_positions}

    all_symbols = set(fid_map.keys()) | set(ai_map.keys())

    matches = []
    fidelity_only = []
    ai_only = []
    discrepancies = []

    for symbol in sorted(all_symbols):
        fid = fid_map.get(symbol)
        ai = ai_map.get(symbol)

        if fid and ai:
            fid_qty = fid.get("quantity", 0)
            ai_qty = ai.get("shares", 0)

            if abs(fid_qty - ai_qty) < 0.01:
                matches.append({
                    "symbol": symbol,
                    "fidelity_shares": fid_qty,
                    "ai_shares": ai_qty,
                    "fidelity_value": fid.get("current_value"),
                    "ai_value": ai.get("current_value"),
                    "fidelity_gain_pct": fid.get("total_gain_loss_pct"),
                    "ai_gain_pct": ai.get("gain_loss_pct"),
                    "ai_score": ai.get("current_score"),
                    "status": "match",
                })
            else:
                discrepancies.append({
                    "symbol": symbol,
                    "fidelity_shares": fid_qty,
                    "ai_shares": ai_qty,
                    "share_diff": round(fid_qty - ai_qty, 6),
                    "fidelity_value": fid.get("current_value"),
                    "ai_value": ai.get("current_value"),
                    "ai_score": ai.get("current_score"),
                    "status": "quantity_mismatch",
                })
        elif fid:
            fidelity_only.append({
                "symbol": symbol,
                "shares": fid.get("quantity"),
                "current_value": fid.get("current_value"),
                "gain_pct": fid.get("total_gain_loss_pct"),
                "cost_basis": fid.get("average_cost_basis"),
                "status": "fidelity_only",
            })
        else:
            ai_only.append({
                "symbol": symbol,
                "shares": ai.get("shares"),
                "current_value": ai.get("current_value"),
                "gain_pct": ai.get("gain_loss_pct"),
                "score": ai.get("current_score"),
                "status": "ai_only",
            })

    fidelity_total = sum(p.get("current_value") or 0 for p in fidelity_positions)
    ai_total = sum(p.get("current_value") or 0 for p in ai_positions)
    overlap_symbols = set(fid_map.keys()) & set(ai_map.keys())
    total_symbols = len(all_symbols)
    overlap_pct = (len(overlap_symbols) / total_symbols * 100) if total_symbols > 0 else 0

    return {
        "matches": matches,
        "fidelity_only": fidelity_only,
        "ai_only": ai_only,
        "discrepancies": discrepancies,
        "summary": {
            "fidelity_total": round(fidelity_total, 2),
            "ai_total": round(ai_total, 2),
            "overlap_count": len(overlap_symbols),
            "total_unique_symbols": total_symbols,
            "overlap_pct": round(overlap_pct, 1),
        }
    }
