-- Repair: SFBS 2:1 split (effective 2026-08-21) was unadjusted, so the
-- stop-loss on 2026-08-21 sold user 3's 41.657 pre-split shares at the
-- post-split price and booked a phantom -$1,835.42 (trade id 382).
-- A split-aware system would have held 83.314 shares and realized -$14.58.
--
-- This credits the missing half of the sale proceeds and restates the
-- trade row. signal_factors.split_artifact=true excludes the row from the
-- reconciliation stop-loss cohort (see backend/main.py, commit ea4d5d5).
--
-- Run once. Idempotence guard: the reason-suffix check below makes a
-- second run a no-op for the trade row; the cash credit is guarded by
-- the same condition via the CTE.

BEGIN;

WITH fixed AS (
    UPDATE ai_portfolio_trades
       SET shares        = 83.3143443089894,
           cost_basis    = 43.885,
           total_value   = 3641.669989745927,
           realized_gain = -14.580010254073,
           reason        = reason || ' [CORRECTED 2026-08-25: SFBS 2:1 split eff 2026-08-21 was unadjusted; true P&L -0.4%. Cash credited +$1,820.83]',
           signal_factors = (signal_factors::jsonb
                             || '{"split_artifact": true, "gain_pct": -0.4}'::jsonb)::json
     WHERE id = 382
       AND ticker = 'SFBS'
       AND reason NOT LIKE '%CORRECTED 2026-08-25%'
     RETURNING id
)
UPDATE ai_portfolio_config
   SET current_cash = current_cash + 1820.8349948729635
 WHERE user_id = 3
   AND EXISTS (SELECT 1 FROM fixed);

COMMIT;

-- Verify afterwards:
--   SELECT realized_gain, reason FROM ai_portfolio_trades WHERE id = 382;
--   SELECT current_cash FROM ai_portfolio_config WHERE user_id = 3;
