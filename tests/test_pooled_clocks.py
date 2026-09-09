"""
Pooled stop-loss clock + live accounts as vintage samples (2026-09-09).

Both exist because the owner-only primary clock sat at n=2 for six weeks
while other accounts, running the SAME champion config, were generating
stops and months of vintage divergence that nothing read.

The rules that matter here are the EXCLUSIONS:

  * the pooled clock counts only stops executed AFTER its registration date,
    because pooling already-observed stops to reach n>=5 is the post-hoc move
    pre-registration exists to prevent;
  * both populations admit only accounts whose config matches the champion.
    Folding an off-config book into vintage spread would measure strategy
    difference as if it were launch luck -- the exact confound the clock
    exists to quantify.
"""

import os
from datetime import date, datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import (
    init_db, SessionLocal, User,
    AIPortfolioConfig, AIPortfolioSnapshot, AIPortfolioTrade, MarketSnapshot,
)
from backend.routes.admin import compute_experiment_gates

OWNER = 1
SAME_CFG = 99020      # champion config -> counted
OFF_CFG = 99021       # different stop -> excluded

init_db()


def _seed():
    db = SessionLocal()
    try:
        for uid in (OWNER, SAME_CFG, OFF_CFG):
            db.query(AIPortfolioConfig).filter_by(user_id=uid).delete()
            db.query(AIPortfolioTrade).filter_by(user_id=uid).delete()
            db.query(AIPortfolioSnapshot).filter_by(user_id=uid).delete()
            if not db.query(User).filter_by(id=uid).first():
                db.add(User(id=uid, email=f"u{uid}@acct.example.org",
                            display_name=f"acct {uid}", is_active=True,
                            is_admin=False, hashed_password=""))
        db.commit()

        # Champion config on owner + SAME_CFG; OFF_CFG differs on the stop.
        for uid, stop in ((OWNER, 8.0), (SAME_CFG, 8.0), (OFF_CFG, 7.0)):
            db.add(AIPortfolioConfig(
                user_id=uid, starting_cash=25000.0, current_cash=25000.0,
                is_active=True, strategy="nostate_cs_bear",
                min_score_to_buy=72, stop_loss_pct=stop, max_positions=8,
            ))

        def stop_trade(uid, when, pct=-8.0):
            cost, shares = 100.0, 10.0
            db.add(AIPortfolioTrade(
                user_id=uid, ticker="ZZZ", action="SELL", shares=shares,
                price=100.0 + pct, total_value=(100.0 + pct) * shares,
                cost_basis=cost, realized_gain=cost * shares * pct / 100.0,
                reason="STOP LOSS: Down 8.0%", executed_at=when,
            ))

        before = datetime(2026, 7, 1, tzinfo=timezone.utc)   # pre-registration
        after = datetime.now(timezone.utc) + timedelta(seconds=5)  # post

        stop_trade(SAME_CFG, before)     # observational only
        stop_trade(SAME_CFG, after)      # counts
        stop_trade(OFF_CFG, after)       # excluded: off-config

        # Snapshots so the accounts qualify as vintage samples.
        for uid in (OWNER, SAME_CFG, OFF_CFG):
            db.add(AIPortfolioSnapshot(
                user_id=uid,
                timestamp=datetime.now(timezone.utc) - timedelta(days=60),
                date=date.today() - timedelta(days=60),
                total_value=25000.0, cash=25000.0, positions_value=0.0,
                positions_count=0, total_return=0.0, total_return_pct=0.0,
            ))
        for days_ago, price in ((60, 500.0), (0, 520.0)):
            d = date.today() - timedelta(days=days_ago)
            row = db.query(MarketSnapshot).filter_by(date=d).first()
            if row:
                row.spy_price = price
            else:
                db.add(MarketSnapshot(date=d, spy_price=price))
        db.commit()
    finally:
        db.close()


@pytest.fixture(scope="module")
def clocks():
    _seed()
    db = SessionLocal()
    try:
        return compute_experiment_gates(db)["program_clocks"]
    finally:
        db.close()


class TestPooledStopClock:

    def test_is_labelled_secondary_and_does_not_replace_the_primary(self, clocks):
        p = clocks["stop_loss_recheck_pooled"]
        assert p["kind"] == "secondary"
        assert p["registered_on"] == "2026-09-09"
        # Primary still present and still its own population.
        assert "stop_loss_recheck" in clocks

    def test_only_champion_config_accounts_are_pooled(self, clocks):
        ids = clocks["stop_loss_recheck_pooled"]["user_ids"]
        assert SAME_CFG in ids
        assert OFF_CFG not in ids, "an off-config account leaked into the pool"

    def test_counts_only_stops_after_registration(self, clocks):
        p = clocks["stop_loss_recheck_pooled"]
        # SAME_CFG has 2 stops but only 1 is after the registration date.
        assert p["n"] == 1

    def test_prior_stops_reported_but_not_counted(self, clocks):
        p = clocks["stop_loss_recheck_pooled"]
        prior = p["observational_prior"]
        assert prior["n"] >= 2               # includes the pre-registration one
        assert prior["n"] > p["n"]
        assert "NOT proof" in prior["note"]

    def test_no_verdict_below_target(self, clocks):
        p = clocks["stop_loss_recheck_pooled"]
        assert p["target"] == 5
        assert p["n"] < 5
        assert p["verdict"] is None


class TestLiveAccountsAsVintageSamples:

    def test_champion_accounts_appear(self, clocks):
        labels = {s["label"] for s in clocks["vintage_spread"]["stacks"]}
        assert f"live u{SAME_CFG}" in labels
        assert f"live u{OWNER}" in labels

    def test_off_config_account_excluded(self, clocks):
        labels = {s["label"] for s in clocks["vintage_spread"]["stacks"]}
        assert f"live u{OFF_CFG}" not in labels, (
            "an off-config book would be measured as vintage luck when it is "
            "actually a strategy difference"
        )

    def test_live_rows_are_tagged(self, clocks):
        live = [s for s in clocks["vintage_spread"]["stacks"]
                if s.get("kind") == "live_account"]
        assert live, "no live accounts folded into the vintage population"
        for s in live:
            assert s["alpha_pp"] is not None
            assert s["days"] > 0


class TestVintageHorizonCohort:
    """Dispersion grows with elapsed time, so the spread must be taken over
    LIKE horizons.

    Before 2026-09-09 the clock pooled every stack regardless of age and
    reported 24.6pp spread / 8.9pp stdev over a population ranging 7..189
    days. That number looks alarming and cannot be gated on -- most of it is
    horizon mixing, not launch luck.
    """

    @staticmethod
    def _cohort_of(rows):
        # Mirrors backend.routes.admin._vintage_spread's selector.
        for anchor in sorted(rows, key=lambda r: -r["days"]):
            grp = [r for r in rows
                   if r["days"] <= anchor["days"]
                   and anchor["days"] <= r["days"] * 2.0]
            if len(grp) >= 2:
                return grp
        return []

    def test_picks_the_longest_comparable_pair(self):
        rows = [
            {"label": "live u2", "days": 189, "alpha_pp": 20.60},
            {"label": "live u1", "days": 184, "alpha_pp": 12.47},
            {"label": "live u4", "days": 22, "alpha_pp": 1.93},
            {"label": "baseline", "days": 21, "alpha_pp": -4.04},
            {"label": "sep02", "days": 7, "alpha_pp": 1.49},
        ]
        c = self._cohort_of(rows)
        assert [r["label"] for r in c] == ["live u2", "live u1"]
        alphas = [r["alpha_pp"] for r in c]
        assert round(max(alphas) - min(alphas), 2) == 8.13
        # The naive all-stacks figure is 3x larger and is the number this
        # change exists to stop anyone gating on.
        allspread = max(r["alpha_pp"] for r in rows) - min(r["alpha_pp"] for r in rows)
        assert round(allspread, 2) == 24.64

    def test_a_lone_long_stack_does_not_form_a_cohort(self):
        # One 200-day stack plus a 5-day one are not comparable; the selector
        # must fall through rather than pair them.
        rows = [
            {"label": "old", "days": 200, "alpha_pp": 10.0},
            {"label": "new", "days": 5, "alpha_pp": 0.5},
        ]
        assert self._cohort_of(rows) == []

    def test_endpoint_reports_the_cohort_it_used(self, clocks):
        v = clocks["vintage_spread"]
        assert "cohort" in v
        assert set(v["cohort"]) >= {"labels", "days_min", "days_max", "rule"}
        assert "all_stacks_spread_pp" in v
