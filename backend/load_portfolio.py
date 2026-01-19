"""
Load portfolio positions from CSV into the database
"""
import csv
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import SessionLocal, PortfolioPosition, init_db

PORTFOLIO_DATA = [
    {"ticker": "CMPS", "shares": 100, "cost_basis": 7.46},
    {"ticker": "ZETA", "shares": 100, "cost_basis": 15.11},
    {"ticker": "HUMA", "shares": 430, "cost_basis": 5.37},
    {"ticker": "LCTX", "shares": 1000, "cost_basis": 1.69},
    {"ticker": "XYZ", "shares": 9.621, "cost_basis": 77.95},
    {"ticker": "IRWD", "shares": 200, "cost_basis": 4.42},
    {"ticker": "GOSS", "shares": 250, "cost_basis": 2.50},
    {"ticker": "TMQ", "shares": 100, "cost_basis": 5.63},
    {"ticker": "ARM", "shares": 32.082, "cost_basis": 111.10},
    {"ticker": "NANC", "shares": 70, "cost_basis": 38.15},
    {"ticker": "VERI", "shares": 100, "cost_basis": 4.44},
    {"ticker": "STE", "shares": 2.071, "cost_basis": 241.40},
    {"ticker": "ONDS", "shares": 50, "cost_basis": 13.03},
    {"ticker": "CAVA", "shares": 15.767, "cost_basis": 73.40},
]


def load_portfolio():
    """Load portfolio positions into database"""
    init_db()
    db = SessionLocal()

    try:
        loaded = 0
        skipped = 0

        for item in PORTFOLIO_DATA:
            # Check if already exists
            existing = db.query(PortfolioPosition).filter(
                PortfolioPosition.ticker == item["ticker"]
            ).first()

            if existing:
                print(f"  Skipped {item['ticker']} - already exists")
                skipped += 1
                continue

            position = PortfolioPosition(
                ticker=item["ticker"],
                shares=item["shares"],
                cost_basis=item["cost_basis"]
            )
            db.add(position)
            print(f"  Added {item['ticker']}: {item['shares']} shares @ ${item['cost_basis']}")
            loaded += 1

        db.commit()
        print(f"\nLoaded {loaded} positions, skipped {skipped}")

    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("Loading portfolio positions...")
    load_portfolio()
    print("Done!")
