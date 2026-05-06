"""ML Signal Layer API routes."""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database import (
    get_db, BacktestRun, BacktestTrade, MLModel, MLPrediction, User,
)
from backend.auth import get_current_active_user, get_admin_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ml"])


def _get_active_model_metric(db, strategy: str, model_type: str) -> tuple:
    """Get current active model's primary metric for comparison.

    Returns (metric_value, model_version) or (None, None) if no active model.
    """
    active = db.query(MLModel).filter(
        MLModel.strategy == strategy,
        MLModel.status == "active",
    ).order_by(desc(MLModel.activated_at)).first()

    if not active:
        return None, None

    if model_type == "regression":
        return active.spearman, active.version
    else:
        return active.roc_auc, active.version


# Standardized eval-backtest config. Held constant so candidate models are
# always compared on the same window — drift in window changes the apparent
# delta and would falsely accept/reject candidates. 4-year window, $25k seed,
# universe="all", ML config from YAML default (so the candidate is exercised
# under the production gating regime that ai_trader uses live).
EVAL_BACKTEST_START = "2022-01-01"
EVAL_BACKTEST_END = "2026-04-28"
EVAL_BACKTEST_CASH = 25000.0
EVAL_BACKTEST_UNIVERSE = "all"

# Graduation thresholds. A new model must beat the incumbent's eval return by
# at least MIN_RETURN_DELTA_PP, and must not regress Sharpe by more than
# MAX_SHARPE_REGRESSION. Both gates required (AND, not OR) — return-only gates
# can accept reckless models that hit return at the cost of risk-adjusted quality.
MIN_RETURN_DELTA_PP = 5.0
MAX_SHARPE_REGRESSION = 0.10

# Absolute CV-metric floors. Phase 1 of the gate is a sanity filter: did the
# model converge to *something*? Anything passing this floor proceeds to the
# eval gate (Phase 2), which decides activation based on portfolio outcome.
# We deliberately don't compare against the incumbent's stored CV metric
# anymore: v12's stored AUC was inflated by training-pool contamination
# pre-fix, and unreachable by any honest retrain. Any reasonable model on
# this domain hits at least 0.55 AUC / 0.05 Spearman.
ABSOLUTE_CV_FLOOR = {
    "classifier": 0.55,
    "regression": 0.05,
}


def _eval_gate_decision(
    candidate_return,
    candidate_sharpe,
    incumbent_return,
    incumbent_sharpe,
) -> tuple:
    """Pure decision function for the eval-backtest graduation gate.

    Returns (passes, reason). passes=True means activate the candidate;
    reason is a human-readable string suitable for the MLModel.error_message
    field or a log line.

    First-model case: when incumbent has no eval baseline (eval_return_pct
    is None — either because no incumbent exists or its baseline wasn't
    backfilled), candidate auto-passes. The candidate's metrics get stored
    so the next model has something to compare against.

    None-on-candidate is a hard fail — the eval backtest didn't produce
    metrics, can't make an informed decision, default to don't-activate.
    """
    if candidate_return is None or candidate_sharpe is None:
        return False, "candidate eval backtest produced no metrics"
    if incumbent_return is None or incumbent_sharpe is None:
        return True, "no incumbent eval baseline; auto-pass and record"
    return_delta = candidate_return - incumbent_return
    sharpe_delta = candidate_sharpe - incumbent_sharpe
    # Small epsilon to avoid float-arithmetic boundary failures (e.g.
    # 2.04 - 0.10 stored as 1.9400000000000002, then 1.94 - 2.04 = -0.1000...01
    # would fail an exact >= -0.10 check). The tolerance is far below any
    # meaningful difference in return (pp) or Sharpe scale.
    eps = 1e-9
    passes_return = return_delta >= MIN_RETURN_DELTA_PP - eps
    passes_sharpe = sharpe_delta >= -MAX_SHARPE_REGRESSION - eps
    if passes_return and passes_sharpe:
        return True, (
            f"return Δ={return_delta:+.2f}pp (≥+{MIN_RETURN_DELTA_PP}), "
            f"sharpe Δ={sharpe_delta:+.3f} (≥{-MAX_SHARPE_REGRESSION})"
        )
    reasons = []
    if not passes_return:
        reasons.append(f"return Δ={return_delta:+.2f}pp < +{MIN_RETURN_DELTA_PP}")
    if not passes_sharpe:
        reasons.append(f"sharpe Δ={sharpe_delta:+.3f} < {-MAX_SHARPE_REGRESSION}")
    return False, "; ".join(reasons)


def _create_and_run_eval_backtest(
    db, model_path: str, strategy: str, name: str, copy_snapshot_from: int = None
) -> dict:
    """Internal helper: create one BacktestRun configured for eval, optionally
    seed its static-data snapshot from another backtest, run synchronously,
    return metrics or {error: ...}.

    copy_snapshot_from: if set, the candidate's snapshot rows are copied to
    this new run BEFORE BacktestEngine.run() executes. The engine sees
    pre-populated rows, skips its own snapshot creation step (idempotent),
    so the run reads from the copied data — same inputs as the source.
    """
    from datetime import date as date_cls
    from backend.database import BacktestRun
    from backend.backtester import run_backtest, copy_backtest_static_snapshot

    start = date_cls.fromisoformat(EVAL_BACKTEST_START)
    end = date_cls.fromisoformat(EVAL_BACKTEST_END)
    bt = BacktestRun(
        name=name,
        status="pending",
        start_date=start,
        end_date=end,
        starting_cash=EVAL_BACKTEST_CASH,
        stock_universe=EVAL_BACKTEST_UNIVERSE,
        strategy=strategy,
        profile_overrides={"ml_signal": {"model_path": model_path}},
    )
    db.add(bt)
    db.commit()
    db.refresh(bt)
    logger.info(f"Eval backtest {bt.id} created ({name}, model={model_path})")

    if copy_snapshot_from is not None:
        copied = copy_backtest_static_snapshot(db, copy_snapshot_from, bt.id)
        logger.info(
            f"Eval backtest {bt.id} seeded with {copied} snapshot rows "
            f"from backtest {copy_snapshot_from} (dual-run: identical inputs)"
        )

    completed = run_backtest(db, bt.id, profile_overrides={"ml_signal": {"model_path": model_path}})
    if completed is None or completed.status != "completed":
        return {
            "error": f"Eval backtest did not complete (status={completed.status if completed else 'unknown'})",
            "backtest_id": bt.id,
        }

    return {
        "backtest_id": completed.id,
        "return_pct": float(completed.total_return_pct) if completed.total_return_pct is not None else None,
        "sharpe": float(completed.sharpe_ratio) if completed.sharpe_ratio is not None else None,
        "max_drawdown_pct": float(completed.max_drawdown_pct) if completed.max_drawdown_pct is not None else None,
    }


def _run_evaluation_backtest(
    db, candidate_path: str, strategy: str, ml_record_id: int,
    incumbent_path: str = None,
) -> dict:
    """Run a standardized backtest with the candidate model file as the
    ML override. Returns {return_pct, sharpe, max_drawdown_pct, backtest_id}
    on success or {error: ...} on failure.

    Synchronous: runs in the calling background task. ~12-15 min per backtest.

    incumbent_path: when provided, performs a DUAL-RUN. The candidate runs
    first (creating a static-data snapshot from current cache state), then
    the incumbent runs against a COPY of the candidate's snapshot — same
    inputs, isolating the model delta from cache drift. Without this, two
    eval backtests minutes apart can drift up to ~12pp on identical models
    (canslim-livescan-churn-investigation.md). Returns separate
    incumbent_* fields when dual-run is performed; the gate compares against
    these instead of the incumbent's stored eval baseline.

    incumbent_path=None: legacy single-run mode. The candidate is compared
    against whatever incumbent.eval_return_pct / eval_sharpe is stored on
    the active MLModel row — subject to cache-drift bias.
    """
    try:
        candidate_result = _create_and_run_eval_backtest(
            db, candidate_path, strategy,
            name=f"eval_for_ml_v{ml_record_id}",
        )
        if "error" in candidate_result:
            return candidate_result

        if incumbent_path is None:
            return candidate_result

        incumbent_result = _create_and_run_eval_backtest(
            db, incumbent_path, strategy,
            name=f"eval_for_ml_v{ml_record_id}_incumbent_baseline",
            copy_snapshot_from=candidate_result["backtest_id"],
        )
        if "error" in incumbent_result:
            # Candidate ran fine; the incumbent control failed. Without a
            # comparable incumbent metric we can't make a dual-run decision.
            # Return the candidate metrics PLUS the failure so the caller
            # can choose between fail-closed (what the gate does) and
            # fall-back-to-stored-baseline (future option).
            return {
                **candidate_result,
                "incumbent_error": incumbent_result["error"],
                "incumbent_backtest_id": incumbent_result.get("backtest_id"),
            }

        # Both legs succeeded — return both sides for the gate.
        return {
            **candidate_result,
            "incumbent_backtest_id": incumbent_result["backtest_id"],
            "incumbent_return_pct": incumbent_result["return_pct"],
            "incumbent_sharpe": incumbent_result["sharpe"],
            "incumbent_max_drawdown_pct": incumbent_result["max_drawdown_pct"],
        }
    except Exception as e:
        logger.error(f"Eval backtest failed for ml v{ml_record_id}: {e}", exc_info=True)
        return {"error": str(e)[:500]}


def _run_training(db_url: str, strategy: str, backtest_ids: list, ml_model_id: int,
                  mode: str = "regression",
                  excluded_features: Optional[list] = None,
                  calibrate: bool = False,
                  auto_activate: bool = True,
                  min_gain_pct: float = 0.0):
    """Background task: extract features, train model, update DB record.

    mode: 'classifier', 'regression', or 'both' (both trains both, saves regression if it passes).

    excluded_features: list of FEATURE_COLUMNS names to drop. When set, this is a
    diagnostic/leakage-audit run — saved to an experimental file, never auto-activated.

    calibrate: wrap the classifier in CalibratedClassifierCV(isotonic) before saving.
    Activates iff it beats the current active model on the primary metric.

    auto_activate: when False, the model is saved to disk + DB but never replaces
    the production active model — used for experimental ablations.

    Includes improvement gate: new model must be >= current active model's
    primary metric to be activated. Models that pass the absolute gate but
    are worse than the incumbent are saved as 'completed' (not 'active').
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        from ml.feature_extractor import extract_training_data, extract_combined_training_data
        from ml.trainer import train_model, train_model_regression, save_model

        ml_record = db.get(MLModel, ml_model_id)
        if not ml_record:
            return

        # Extract training data: backtest trades + live trades (combined)
        if backtest_ids:
            # Explicit backtest IDs — use original extraction only
            result_tuple = extract_training_data(db, strategy=strategy, backtest_ids=backtest_ids)
        else:
            # Auto-select: combine backtest + live trades for maximum data
            result_tuple = extract_combined_training_data(db, strategy=strategy, include_live=True)
        df, dedup_stats = result_tuple

        if df.empty:
            ml_record.status = "failed"
            ml_record.error_message = "No training data extracted"
            db.commit()
            return

        ml_record.training_samples = len(df)
        db.commit()

        logger.info(
            f"Training data: {dedup_stats['trades_after_dedup']} trades "
            f"(from {dedup_stats['backtests_after']}/{dedup_stats['backtests_before']} backtests, "
            f"{dedup_stats['trades_before_dedup'] - dedup_stats['trades_after_dedup']} duplicates removed)"
        )

        # Train based on mode. Experimental kwargs only apply to classifier path
        # — regression doesn't have isotonic calibration in this trainer.
        result = None
        if mode == "classifier":
            result = train_model(df, excluded_features=excluded_features, calibrate=calibrate, min_gain_pct=min_gain_pct)
        elif mode == "regression":
            result = train_model_regression(df)
        elif mode == "both":
            # Train both, prefer regression if it passes
            cls_result = train_model(df, excluded_features=excluded_features, calibrate=calibrate, min_gain_pct=min_gain_pct)
            reg_result = train_model_regression(df)
            if reg_result.get("passed_gate"):
                result = reg_result
                logger.info("Both models trained — using regression (passed gate)")
            elif cls_result.get("passed_gate"):
                result = cls_result
                logger.info("Both models trained — using classifier (regression failed gate)")
            else:
                # Neither passed — report regression failure (more informative)
                result = reg_result
        else:
            result = train_model_regression(df)

        model_type = result.get("model_type", "classifier")

        if not result.get("passed_gate"):
            ml_record.status = "failed"
            ml_record.model_type = model_type
            ml_record.error_message = result.get("error", f"Model failed {'Spearman' if model_type == 'regression' else 'ROC AUC'} gate")
            if model_type == "regression":
                ml_record.spearman = result.get("mean_spearman")
            else:
                ml_record.roc_auc = result.get("mean_roc_auc")
            ml_record.cv_results = result.get("cv_results")
            db.commit()
            return

        # Save model to disk. Experimental runs (excluded_features set, or
        # auto_activate=False) save to a versioned experimental path so they
        # cannot accidentally overwrite the active production model.
        model = result["model"]
        metrics = result["metrics"]
        is_experimental = bool(excluded_features) or not auto_activate
        # CRITICAL: must be training order, not importance-sorted. xgboost rejects
        # predict() with feature_names_mismatch when order differs (v17 hit this in
        # both live + OOS paths — commits 1bbf4c9 + 61d55fb worked around it). Trainer
        # now emits result["feature_columns"] in training order; older paths that
        # read feature_importance.keys() got the sorted-by-importance list, which is
        # the bug.
        active_features = result.get("feature_columns") or None
        from ml.trainer import MODEL_DIR
        if is_experimental:
            exp_path = MODEL_DIR / f"ml_model_v{ml_record.version}_experimental.joblib"
            model_path = save_model(model, {
                "strategy": strategy,
                "version": ml_record.version,
                "training_samples": result["training_samples"],
                "feature_importance": result["feature_importance"],
                "model_type": model_type,
                "excluded_features": list(excluded_features or []),
                "calibrated": bool(calibrate),
                "experimental": True,
            }, path=exp_path, feature_columns=active_features)
        else:
            # CRITICAL: save to versioned candidate path, NOT directly to
            # ACTIVE_MODEL_PATH. Otherwise the eval gate's incumbent file
            # gets overwritten by the candidate before the gate runs, and
            # the comparison ends up "candidate vs candidate". Promotion
            # to active.joblib happens only after the gate passes (see
            # _promote_to_active_path below).
            candidate_path = MODEL_DIR / f"ml_model_v{ml_record.version}.joblib"
            model_path = save_model(model, {
                "strategy": strategy,
                "version": ml_record.version,
                "training_samples": result["training_samples"],
                "feature_importance": result["feature_importance"],
                "model_type": model_type,
                "calibrated": bool(calibrate),
            }, path=candidate_path, feature_columns=active_features)

        # Update DB record with metrics
        ml_record.model_type = model_type

        if model_type == "regression":
            new_metric = metrics.get("spearman")
            ml_record.spearman = new_metric
            ml_record.r2_score = metrics.get("r2")
            ml_record.mae = metrics.get("mae")
            ml_record.direction_accuracy = metrics.get("direction_accuracy")
            metric_name = "Spearman"
        else:
            new_metric = metrics.get("roc_auc")
            ml_record.roc_auc = new_metric
            ml_record.accuracy = metrics.get("accuracy")
            ml_record.precision_score = metrics.get("precision")
            ml_record.recall_score = metrics.get("recall")
            ml_record.f1 = metrics.get("f1")
            ml_record.brier_score = metrics.get("brier_score")
            metric_name = "ROC AUC"

        ml_record.cv_results = result.get("cv_results")
        ml_record.feature_importance = result.get("feature_importance")
        ml_record.feature_count = result.get("feature_count")
        ml_record.model_path = str(model_path)
        ml_record.training_samples = result.get("training_samples")

        # Experimental runs (leakage audit, calibrated holdout) never replace
        # the active model. They land as 'completed' so the metrics are
        # comparable in the model list, but production keeps using the
        # incumbent until we explicitly graduate one via /api/ml/promote.
        if is_experimental:
            ml_record.status = "completed"
            tag = []
            if excluded_features:
                tag.append(f"excluded={','.join(excluded_features)}")
            if calibrate:
                tag.append("calibrated")
            if not auto_activate:
                tag.append("auto_activate=false")
            ml_record.error_message = "Experimental run: " + "; ".join(tag)
            db.commit()
            logger.info(f"Experimental v{ml_record.version} saved (not activated): {tag}")
            return

        # Minimum sample count gate: don't activate until enough data
        from ml.trainer import MIN_TOTAL_SAMPLES
        training_samples = result.get("training_samples", 0)
        if training_samples < MIN_TOTAL_SAMPLES:
            ml_record.status = "completed"
            ml_record.error_message = (
                f"Data accumulation: {training_samples}/{MIN_TOTAL_SAMPLES} samples. "
                f"Model saved but not activated."
            )
            db.commit()
            logger.info(
                f"ML model v{ml_record.version} in accumulation mode: "
                f"{training_samples}/{MIN_TOTAL_SAMPLES} samples needed"
            )
            return

        # Phase 1: absolute CV-metric floor. The May 5 OOS diagnostic proved
        # AUC is the wrong selection metric for this strategy (top-decile WR
        # matters, not full-distribution rank correctness), so the eval gate
        # at Phase 2 is the authoritative criterion. Phase 1's job is just to
        # filter out garbage models that didn't converge — not to compare
        # against incumbent CV. Comparing against incumbent CV was over-blocking:
        # v12's stored AUC (0.6116) was inflated by training-pool contamination
        # before the Apr 29 fix, and every cleaner retrain (~0.5877) was
        # auto-rejected before the eval gate could even fire. Switched to
        # absolute floors: garbage filter only, eval gate decides activation.
        if new_metric is not None and new_metric < ABSOLUTE_CV_FLOOR[model_type]:
            ml_record.status = "completed"
            ml_record.error_message = (
                f"Not activated: {metric_name} {new_metric:.4f} < absolute floor "
                f"{ABSOLUTE_CV_FLOOR[model_type]:.4f}"
            )
            db.commit()
            logger.warning(
                f"ML model v{ml_record.version} blocked at absolute floor: "
                f"{metric_name} {new_metric:.4f} < {ABSOLUTE_CV_FLOOR[model_type]:.4f}"
            )
            return

        # Log the historical incumbent comparison for diagnostic context only.
        # No longer used as a blocking gate — see comment above.
        current_metric, current_version = _get_active_model_metric(db, strategy, model_type)
        if current_metric is not None and new_metric is not None:
            delta = new_metric - current_metric
            logger.info(
                f"ML v{ml_record.version} CV {metric_name}={new_metric:.4f} "
                f"vs incumbent v{current_version} {current_metric:.4f} (Δ={delta:+.4f}, informational)"
            )

        # Improvement gate (Phase 2): standardized eval backtest. AUC/Spearman
        # were shown to disagree with portfolio return (May 5 diagnostic) so
        # gating on CV alone is insufficient.
        #
        # Dual-run pattern: when an incumbent has an eval baseline, run BOTH
        # the candidate and the incumbent against the SAME static-data
        # snapshot — eliminates cache-drift bias (without it, a candidate
        # measured today can falsely beat an incumbent baseline measured
        # last week purely because the cache shifted in between, ~12pp
        # observed). First-model case (no incumbent baseline): single-run
        # the candidate, store as new baseline, auto-activate.
        incumbent = db.query(MLModel).filter(
            MLModel.strategy == strategy,
            MLModel.status == "active",
            MLModel.id != ml_record.id,
        ).order_by(desc(MLModel.activated_at)).first()
        incumbent_path = (
            incumbent.model_path if incumbent is not None and incumbent.model_path else None
        )
        do_dual_run = incumbent is not None and incumbent_path is not None

        logger.info(
            f"Running eval backtest for ml v{ml_record.version} candidate "
            f"({'dual-run vs v' + str(incumbent.version) if do_dual_run else 'single-run, no incumbent baseline'})"
        )
        eval_result = _run_evaluation_backtest(
            db, str(model_path), strategy, ml_record.id,
            incumbent_path=incumbent_path if do_dual_run else None,
        )

        if "error" in eval_result:
            ml_record.status = "failed"
            ml_record.error_message = f"Eval backtest failed: {eval_result['error']}"[:500]
            db.commit()
            logger.error(
                f"ML v{ml_record.version} eval backtest failed: {eval_result.get('error')}"
            )
            return

        ml_record.eval_backtest_id = eval_result.get("backtest_id")
        ml_record.eval_return_pct = eval_result.get("return_pct")
        ml_record.eval_sharpe = eval_result.get("sharpe")
        ml_record.eval_max_drawdown_pct = eval_result.get("max_drawdown_pct")
        db.commit()

        # Decide which incumbent metrics to compare against:
        #  - dual-run succeeded: use the freshly-computed incumbent_* values
        #    (apples-to-apples on the same snapshot — no drift)
        #  - dual-run failed mid-flight (incumbent_error): fall back to the
        #    stored eval_return_pct on the incumbent row, with the caveat
        #    documented in MEMORY.md (drift-biased but better than nothing).
        #  - single-run (first-model bootstrap): incumbent metrics are None
        #    and _eval_gate_decision auto-passes.
        if do_dual_run and "incumbent_return_pct" in eval_result:
            incumbent_return = eval_result["incumbent_return_pct"]
            incumbent_sharpe = eval_result["incumbent_sharpe"]
            comparison_mode = "dual-run (same snapshot)"
        elif do_dual_run and "incumbent_error" in eval_result:
            incumbent_return = incumbent.eval_return_pct
            incumbent_sharpe = incumbent.eval_sharpe
            comparison_mode = f"single-run fallback (incumbent leg failed: {eval_result['incumbent_error']})"
            logger.warning(
                f"ML v{ml_record.version}: dual-run incumbent leg failed, "
                f"falling back to stored baseline (drift-biased)"
            )
        else:
            incumbent_return = None
            incumbent_sharpe = None
            comparison_mode = "first-model (no incumbent baseline)"

        passes, reason = _eval_gate_decision(
            ml_record.eval_return_pct, ml_record.eval_sharpe,
            incumbent_return, incumbent_sharpe,
        )
        incumbent_descr = (
            f"incumbent v{incumbent.version}" if incumbent is not None else "no incumbent"
        )
        candidate_descr = (
            f"ret={ml_record.eval_return_pct:.2f}% sh={ml_record.eval_sharpe:.3f}"
            if ml_record.eval_return_pct is not None and ml_record.eval_sharpe is not None
            else "ret=? sh=?"
        )
        if not passes:
            ml_record.status = "completed"
            ml_record.error_message = (
                f"Eval gate ({comparison_mode}): {reason}. "
                f"Candidate: {candidate_descr} vs {incumbent_descr}"
            )[:500]
            db.commit()
            logger.warning(
                f"ML v{ml_record.version} blocked by eval gate: {reason} ({comparison_mode})"
            )
            return
        logger.info(
            f"ML v{ml_record.version} passed eval gate ({reason}); "
            f"comparison_mode={comparison_mode}; candidate {candidate_descr} vs {incumbent_descr}"
        )

        # Promote candidate file to ACTIVE_MODEL_PATH. Done BEFORE the DB
        # status flip so a copy failure doesn't leave us with an "active"
        # row pointing at a stale file. ml_record.model_path stays at the
        # versioned path (immutable training output); the active path is a
        # copy that the live process loads via reload_model. This separation
        # is what keeps the eval gate's incumbent file intact during gating.
        try:
            import shutil
            from ml.trainer import ACTIVE_MODEL_PATH
            shutil.copy2(str(model_path), str(ACTIVE_MODEL_PATH))
            logger.info(
                f"Promoted v{ml_record.version} candidate {model_path} → active path {ACTIVE_MODEL_PATH}"
            )
        except Exception as e:
            ml_record.status = "failed"
            ml_record.error_message = f"Promotion to active path failed: {e}"[:500]
            db.commit()
            logger.error(f"ML v{ml_record.version} promotion failed: {e}", exc_info=True)
            return

        # Deactivate previous active models for this strategy
        # Exclude current record to avoid self-deactivation race
        db.query(MLModel).filter(
            MLModel.strategy == strategy,
            MLModel.status == "active",
            MLModel.id != ml_record.id,
        ).update({"status": "completed"})

        ml_record.status = "active"
        ml_record.activated_at = datetime.now(timezone.utc)
        db.commit()

        # Reload model in memory — picks up the freshly-copied active.joblib
        try:
            from ml.model import reload_model
            reload_model()
        except Exception:
            pass

        primary_metric = f"{metric_name}={new_metric:.4f}" if new_metric else "unknown"
        logger.info(f"ML model v{ml_record.version} ({model_type}) trained and activated: {primary_metric}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        try:
            ml_record = db.get(MLModel, ml_model_id)
            if ml_record:
                ml_record.status = "failed"
                ml_record.error_message = str(e)[:500]
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


@router.post("/train")
async def trigger_training(
    background_tasks: BackgroundTasks,
    strategy: str = Query(default="nostate_optimized"),
    backtest_ids: str = Query(default="", description="Comma-separated backtest IDs (empty=all)"),
    mode: str = Query(default="regression", description="Training mode: classifier, regression, or both"),
    excluded_features: str = Query(default="", description="Comma-separated feature names to drop (leakage audit). Forces experimental save."),
    calibrate: bool = Query(default=False, description="Wrap classifier in CalibratedClassifierCV(isotonic). Required for min_confidence to mean a real probability."),
    auto_activate: bool = Query(default=True, description="If false, model is saved + recorded but never replaces the active model."),
    min_gain_pct: float = Query(default=0.0, ge=0.0, description="Classifier label threshold: positive class = gain_pct > min_gain_pct. Default 0.0 = 'any winner'. v12 used 10.0 ('big winners only') — different task, different (higher) AUC. Classifier mode only."),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Trigger ML model training (admin only). Runs in background.

    Experimental flags (excluded_features, auto_activate=false) save the model
    to ml_model_v{N}_experimental.joblib and mark its DB row 'completed' —
    production keeps the current active model.
    """
    if mode not in ("classifier", "regression", "both"):
        raise HTTPException(400, "mode must be 'classifier', 'regression', or 'both'")
    excluded_list = [f.strip() for f in excluded_features.split(",") if f.strip()]

    # Concurrent training guard: prevent overlapping training runs
    in_progress = db.query(MLModel).filter(
        MLModel.strategy == strategy,
        MLModel.status == "training",
    ).first()
    if in_progress:
        # Check if it's a stale training run (> 30 min)
        if in_progress.created_at and (
            datetime.now(timezone.utc) - in_progress.created_at.replace(tzinfo=timezone.utc)
            < timedelta(minutes=30)
        ):
            raise HTTPException(
                409,
                f"Training already in progress: v{in_progress.version} "
                f"(started {in_progress.created_at.isoformat()}). "
                f"Wait for it to complete or check /api/ml/health for stuck runs."
            )
        else:
            # Stale training run — mark as failed and proceed
            in_progress.status = "failed"
            in_progress.error_message = "Timed out (stale training run detected)"
            db.commit()
            logger.warning(f"Marked stale training run v{in_progress.version} as failed")

    # Parse backtest_ids
    ids = []
    if backtest_ids.strip():
        try:
            ids = [int(x.strip()) for x in backtest_ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "Invalid backtest_ids format")

    # Get next version number
    latest = db.query(MLModel).filter(MLModel.strategy == strategy).order_by(desc(MLModel.version)).first()
    next_version = (latest.version + 1) if latest else 1

    # Create DB record
    ml_record = MLModel(
        version=next_version,
        strategy=strategy,
        status="training",
        model_type=mode if mode != "both" else "regression",
        backtest_ids=ids or None,
        hyperparameters={
            "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
            "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 1.0, "reg_lambda": 5.0,
            "mode": mode,
            "min_gain_pct": min_gain_pct,
        },
    )
    db.add(ml_record)
    db.commit()
    db.refresh(ml_record)

    # Get DB URL for background task (needs its own session)
    from backend.database import DATABASE_URL
    background_tasks.add_task(
        _run_training, DATABASE_URL, strategy, ids, ml_record.id, mode,
        excluded_list or None, calibrate, auto_activate, min_gain_pct,
    )

    return {
        "message": f"Training started for v{next_version} ({mode})",
        "model_id": ml_record.id,
        "version": next_version,
        "strategy": strategy,
        "mode": mode,
        "excluded_features": excluded_list,
        "calibrate": calibrate,
        "auto_activate": auto_activate,
        "experimental": bool(excluded_list) or not auto_activate,
    }


@router.post("/evaluate-oos")
async def evaluate_oos_endpoint(
    strategy: str = Query(default="nostate_optimized"),
    cutoff_iso: Optional[str] = Query(default=None, description="ISO timestamp; backtests created on/after are the OOS holdout. Defaults to (now - holdout_hours)."),
    holdout_hours: float = Query(default=72.0, description="Used when cutoff_iso is omitted: cutoff = now - holdout_hours."),
    model_ids: str = Query(default="", description="Comma-separated MLModel.id values to compare. Empty = current active model only."),
    min_gain_pct: float = Query(default=10.0, description="Label threshold: positive class = gain_pct > min_gain_pct."),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Evaluate one or more saved models on backtests the model never saw.

    Distinct from walk-forward CV — OOS uses backtests created AFTER the
    cutoff, which (assuming the model was trained before cutoff) the model
    cannot have learned from. This is the only way to compare candidate
    models honestly: training metrics from different models trained on
    different (often overlapping) pools aren't comparable.

    Caveats:
    - Caller is responsible for picking a cutoff that places each candidate
      model's training data on the correct side. There's no automatic check
      that model X was actually trained pre-cutoff.
    - The model file must exist on disk (joblib payload). MLModel.model_path
      is consulted; missing rows are reported per-id.
    """
    from ml.oos_eval import compare_models_oos, evaluate_oos

    if cutoff_iso:
        try:
            cutoff = datetime.fromisoformat(cutoff_iso.replace("Z", "+00:00"))
        except ValueError as e:
            raise HTTPException(400, f"Invalid cutoff_iso: {e}")
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=holdout_hours)

    # Resolve cutoff to naive UTC for comparison with DB columns (which are naive)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.astimezone(timezone.utc).replace(tzinfo=None)

    # Build {label: model_path} mapping
    ids = [int(x) for x in model_ids.split(",") if x.strip()]
    paths = {}
    if ids:
        rows = db.query(MLModel).filter(MLModel.id.in_(ids)).all()
        for r in rows:
            if not r.model_path:
                continue
            paths[f"v{r.version}_id{r.id}"] = r.model_path
        missing = set(ids) - {r.id for r in rows}
        if missing:
            raise HTTPException(404, f"MLModel ids not found: {sorted(missing)}")
    else:
        # Default: current active model
        active = db.query(MLModel).filter(
            MLModel.strategy == strategy,
            MLModel.status == "active",
        ).order_by(desc(MLModel.id)).first()
        if not active or not active.model_path:
            raise HTTPException(404, f"No active model with stored path for {strategy}")
        paths[f"v{active.version}_active"] = active.model_path

    if len(paths) == 1:
        # Single model — return its metrics directly for cleaner JSON
        label, path = next(iter(paths.items()))
        result = evaluate_oos(db, path, strategy, cutoff, min_gain_pct=min_gain_pct)
        return {"label": label, "result": result, "cutoff_used": cutoff.isoformat()}

    return compare_models_oos(db, paths, strategy, cutoff, min_gain_pct=min_gain_pct)


@router.post("/diagnose")
async def diagnose_models(
    model_ids: str = Query(..., description="Comma-separated MLModel.id list, e.g. '12,17,18'"),
    strategy: str = Query(default="nostate_optimized"),
    cutoff_iso: str = Query(..., description="ISO datetime; backtests created on/after are the OOS holdout."),
    min_gain_pct: float = Query(default=10.0, description="Label threshold: positive class = gain_pct > min_gain_pct."),
    threshold: float = Query(default=0.30, description="Pick threshold for the disagreement breakdown (production veto threshold = 0.30)."),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Per-trade ML model diagnostics on an OOS holdout slice.

    Computes (a) probability distribution per model, (b) pairwise Spearman
    rank correlation, (c) disagreement at threshold (only-A / only-B / both
    / neither, with WRs), (d) top-decile WR, (e) bottom-decile WR.

    Replaces the VPS-only oos_disagreement.py one-off script. Used to
    investigate why a higher-AUC candidate doesn't always translate into a
    backtest-return advantage — the disagreement structure typically shows
    the AUC delta concentrated in middle-rank discrimination that
    backtest's score+max_positions filter never reaches.

    Looks up MLModel.model_path regardless of status — works for active,
    completed, and experimental models alike.
    """
    from ml.diagnostics import run_full_diagnostic
    from ml.oos_eval import get_holdout_trades

    try:
        cutoff = datetime.fromisoformat(cutoff_iso.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(400, f"Invalid cutoff_iso: {e}")
    if cutoff.tzinfo is not None:
        cutoff = cutoff.astimezone(timezone.utc).replace(tzinfo=None)

    try:
        ids = [int(x.strip()) for x in model_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "Invalid model_ids format (expected comma-separated integers)")
    if not ids:
        raise HTTPException(400, "model_ids must contain at least one ID")

    rows = db.query(MLModel).filter(MLModel.id.in_(ids)).all()
    by_id = {r.id: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise HTTPException(404, f"MLModel ids not found: {missing}")

    paths: dict = {}
    no_path: list = []
    for mid in ids:
        r = by_id[mid]
        if not r.model_path:
            no_path.append(mid)
            continue
        paths[f"v{r.version}_id{r.id}"] = r.model_path
    if no_path:
        raise HTTPException(
            400,
            f"MLModel ids missing model_path (no saved file): {no_path}",
        )
    if len(paths) < 2:
        raise HTTPException(
            400,
            "Diagnostic needs at least 2 models with saved paths to be meaningful",
        )

    holdout = get_holdout_trades(db, strategy, cutoff, after=True)
    if holdout.empty:
        return {
            "error": f"No holdout trades for {strategy} after {cutoff.isoformat()}",
            "cutoff_used": cutoff.isoformat(),
            "strategy": strategy,
            "models": [{"id": mid, "version": by_id[mid].version, "model_path": by_id[mid].model_path} for mid in ids],
        }

    diagnostic = run_full_diagnostic(
        paths, holdout, min_gain_pct=min_gain_pct, threshold=threshold,
    )
    diagnostic["cutoff_used"] = cutoff.isoformat()
    diagnostic["strategy"] = strategy
    diagnostic["holdout_backtest_count"] = (
        int(holdout["backtest_id"].nunique()) if "backtest_id" in holdout.columns else None
    )
    return diagnostic


@router.get("/health")
async def get_ml_health(
    strategy: str = Query(default="nostate_optimized"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """ML system health check — surfaces warnings about model quality and data integrity."""
    warnings = []

    # Check 1: Only one active model per strategy
    active_count = db.query(MLModel).filter(
        MLModel.strategy == strategy,
        MLModel.status == "active",
    ).count()

    if active_count == 0:
        warnings.append({"level": "error", "message": "No active model — predictions disabled"})
    elif active_count > 1:
        warnings.append({
            "level": "error",
            "message": f"{active_count} active models found (expected 1) — run /api/ml/fix-active to resolve",
        })

    # Check 2: Active model quality
    active = db.query(MLModel).filter(
        MLModel.strategy == strategy,
        MLModel.status == "active",
    ).order_by(desc(MLModel.activated_at)).first()

    model_info = None
    if active:
        model_info = {
            "version": active.version,
            "model_type": active.model_type or "classifier",
            "activated_at": active.activated_at.isoformat() if active.activated_at else None,
        }
        if (active.model_type or "classifier") == "regression":
            if active.spearman is not None and active.spearman < 0.10:
                warnings.append({
                    "level": "warning",
                    "message": f"Active model Spearman {active.spearman:.4f} below 0.10 gate",
                })
            model_info["spearman"] = active.spearman
        else:
            if active.roc_auc is not None and active.roc_auc < 0.52:
                warnings.append({
                    "level": "warning",
                    "message": f"Active model ROC AUC {active.roc_auc:.4f} below 0.52 gate",
                })
            if active.roc_auc is not None and active.roc_auc < 0.50:
                warnings.append({
                    "level": "error",
                    "message": f"Active model ROC AUC {active.roc_auc:.4f} is anti-predictive (<0.50)",
                })
            model_info["roc_auc"] = active.roc_auc

    # Check 3: Stuck training runs
    stuck = db.query(MLModel).filter(
        MLModel.strategy == strategy,
        MLModel.status == "training",
    ).all()
    for s in stuck:
        age = None
        if s.created_at:
            age = (datetime.now(timezone.utc) - s.created_at.replace(tzinfo=timezone.utc)).total_seconds()
        if age and age > 1800:  # 30 min
            warnings.append({
                "level": "warning",
                "message": f"Stuck training run: v{s.version} started {int(age // 60)}m ago",
            })

    # Check 4: Model loaded in memory
    try:
        from ml.model import is_model_loaded
        loaded = is_model_loaded()
    except ImportError:
        loaded = False
    if not loaded and active_count > 0:
        warnings.append({"level": "warning", "message": "Active model exists but not loaded in memory"})

    # Check 5: All models summary
    total_models = db.query(MLModel).filter(MLModel.strategy == strategy).count()
    failed_models = db.query(MLModel).filter(
        MLModel.strategy == strategy, MLModel.status == "failed",
    ).count()

    return {
        "healthy": len([w for w in warnings if w["level"] == "error"]) == 0,
        "warnings": warnings,
        "active_model": model_info,
        "model_loaded_in_memory": loaded,
        "total_models": total_models,
        "active_count": active_count,
        "failed_count": failed_models,
    }


@router.post("/fix-active")
async def fix_active_models(
    strategy: str = Query(default="nostate_optimized"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Fix multiple active models — keep only the best one (admin only)."""
    active_models = db.query(MLModel).filter(
        MLModel.strategy == strategy,
        MLModel.status == "active",
    ).all()

    if len(active_models) <= 1:
        return {"message": "No fix needed — 0 or 1 active models", "active_count": len(active_models)}

    # Find the best model by primary metric
    best = None
    for m in active_models:
        if best is None:
            best = m
            continue
        best_metric = best.spearman if (best.model_type or "classifier") == "regression" else best.roc_auc
        m_metric = m.spearman if (m.model_type or "classifier") == "regression" else m.roc_auc
        if m_metric is not None and (best_metric is None or m_metric > best_metric):
            best = m

    # Deactivate all except the best
    deactivated = []
    for m in active_models:
        if m.id != best.id:
            m.status = "completed"
            deactivated.append(f"v{m.version} (id={m.id})")

    db.commit()

    return {
        "message": f"Fixed: kept v{best.version} (id={best.id}) active, deactivated {len(deactivated)} others",
        "kept_active": {"version": best.version, "id": best.id},
        "deactivated": deactivated,
    }


@router.get("/status")
async def get_ml_status(
    strategy: str = Query(default="nostate_optimized"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get current ML model status and metrics."""
    # Active model
    active = db.query(MLModel).filter(
        MLModel.strategy == strategy,
        MLModel.status == "active",
    ).order_by(desc(MLModel.activated_at)).first()

    # Latest training run
    latest = db.query(MLModel).filter(
        MLModel.strategy == strategy,
    ).order_by(desc(MLModel.created_at)).first()

    # Check if model is loaded in memory
    try:
        from ml.model import is_model_loaded
        loaded = is_model_loaded()
    except ImportError:
        loaded = False

    # Get config
    try:
        from config_loader import config
        ml_config = config.get(f"strategy_profiles.{strategy}.ml_signal", {})
    except Exception:
        ml_config = {}

    result = {
        "active_model": None,
        "latest_training": None,
        "model_loaded_in_memory": loaded,
        "config": ml_config,
    }

    if active:
        model_info = {
            "id": active.id,
            "version": active.version,
            "model_type": active.model_type or "classifier",
            "training_samples": active.training_samples,
            "feature_count": active.feature_count,
            "activated_at": active.activated_at.isoformat() if active.activated_at else None,
        }
        if (active.model_type or "classifier") == "regression":
            model_info.update({
                "spearman": active.spearman,
                "r2_score": active.r2_score,
                "mae": active.mae,
                "direction_accuracy": active.direction_accuracy,
            })
        else:
            model_info.update({
                "roc_auc": active.roc_auc,
                "accuracy": active.accuracy,
                "f1": active.f1,
            })
        result["active_model"] = model_info

    if latest:
        result["latest_training"] = {
            "id": latest.id,
            "version": latest.version,
            "status": latest.status,
            "error_message": latest.error_message,
            "created_at": latest.created_at.isoformat() if latest.created_at else None,
        }

    return result


@router.get("/predict/{ticker}")
async def predict_ticker(
    ticker: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Debug: get ML prediction for a specific stock using current scan data."""
    ticker = ticker.upper().strip()

    from backend.database import Stock, StockScore
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    if not stock:
        raise HTTPException(404, f"Stock {ticker} not found")

    score = db.query(StockScore).filter(StockScore.ticker == ticker).order_by(desc(StockScore.scored_at)).first()

    try:
        from ml.model import get_ml_prediction
        prediction = get_ml_prediction(
            total_score=stock.total_score or 0,
            composite_score=stock.total_score or 0,  # Use total_score as proxy
            entry_type=2,  # standard (we don't know without full evaluation)
            market_regime=1,  # neutral default
            estimate_revision_bonus=0,
            coiled_spring=0,
            soft_zone=0,
            soft_zone_multiplier=1.0,
            deterministic_boost=0,
        )
    except ImportError:
        prediction = None

    return {
        "ticker": ticker,
        "ml_confidence": prediction,
        "total_score": stock.total_score,
        "note": "Debug endpoint — uses defaults for context-dependent features (entry_type, regime, bonuses)",
    }


@router.get("/features")
async def get_feature_importance(
    strategy: str = Query(default="nostate_optimized"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Feature importance from active model."""
    active = db.query(MLModel).filter(
        MLModel.strategy == strategy,
        MLModel.status == "active",
    ).first()

    if not active:
        raise HTTPException(404, "No active model found")

    return {
        "version": active.version,
        "feature_importance": active.feature_importance or {},
        "training_samples": active.training_samples,
        "roc_auc": active.roc_auc,
    }


@router.get("/validation")
async def get_validation_results(
    strategy: str = Query(default="nostate_optimized"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Walk-forward CV fold details from active model."""
    active = db.query(MLModel).filter(
        MLModel.strategy == strategy,
        MLModel.status == "active",
    ).first()

    if not active:
        raise HTTPException(404, "No active model found")

    resp = {
        "version": active.version,
        "model_type": active.model_type or "classifier",
        "cv_results": active.cv_results or [],
    }
    if (active.model_type or "classifier") == "regression":
        resp.update({
            "spearman": active.spearman,
            "r2_score": active.r2_score,
            "mae": active.mae,
            "direction_accuracy": active.direction_accuracy,
        })
    else:
        resp.update({
            "roc_auc": active.roc_auc,
            "accuracy": active.accuracy,
            "precision": active.precision_score,
            "recall": active.recall_score,
            "f1": active.f1,
            "brier_score": active.brier_score,
        })
    return resp


@router.get("/training-data")
async def preview_training_data(
    strategy: str = Query(default="nostate_optimized"),
    limit: int = Query(default=50, le=500),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Preview extracted training dataset with dedup stats (admin only).
    Now includes live trades combined with backtest trades."""
    try:
        from ml.feature_extractor import extract_combined_training_data
        df, dedup_stats = extract_combined_training_data(db, strategy=strategy, include_live=True)
    except Exception as e:
        raise HTTPException(500, f"Feature extraction failed: {e}")

    if df.empty:
        return {"total": 0, "win_rate": None, "dedup_stats": dedup_stats, "samples": []}

    # Count live vs backtest trades
    live_count = len(df[df["backtest_id"] == -1]) if "backtest_id" in df.columns else 0
    bt_count = len(df) - live_count

    return {
        "total": len(df),
        "backtest_trades": bt_count,
        "live_trades": live_count,
        "win_rate": round(float(df["win"].mean()), 4),
        "mean_gain": round(float(df["gain_pct"].mean()), 2),
        "dedup_stats": dedup_stats,
        "samples": df.head(limit).to_dict(orient="records"),
    }


@router.post("/compare")
async def compare_ml_backtest(
    start_date: date = Query(...),
    end_date: date = Query(...),
    starting_cash: float = Query(25000.0),
    strategy: str = Query("nostate_optimized"),
    stock_universe: str = Query("all"),
    min_confidence: float = Query(0.5, description="ML min_confidence threshold for active run"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Launch paired A/B backtests: ML OFF (baseline) vs ML ACTIVE (with gating).

    Both backtests run on the same data/cache for a fair comparison.
    Results are viewable in the existing backtest comparison UI.
    """
    from backend.backtest_queue import backtest_queue

    # Baseline: ML log_only (no influence on trading)
    baseline = BacktestRun(
        user_id=current_user.id,
        name=f"[ML OFF] {strategy} {start_date}→{end_date}",
        start_date=start_date,
        end_date=end_date,
        starting_cash=starting_cash,
        stock_universe=stock_universe,
        strategy=strategy,
        status="pending",
        profile_overrides={"ml_signal": {"enabled": True, "log_only": True, "min_confidence": 0.0}},
    )
    db.add(baseline)
    db.flush()

    # Active: ML modifies composite_score + confidence gating
    active = BacktestRun(
        user_id=current_user.id,
        name=f"[ML ACTIVE min={min_confidence}] {strategy} {start_date}→{end_date}",
        start_date=start_date,
        end_date=end_date,
        starting_cash=starting_cash,
        stock_universe=stock_universe,
        strategy=strategy,
        status="pending",
        profile_overrides={
            "ml_signal": {
                "enabled": True,
                "log_only": False,
                "min_confidence": min_confidence,
                "veto_action": "skip",
            }
        },
    )
    db.add(active)
    db.commit()
    db.refresh(baseline)
    db.refresh(active)

    # Enqueue both — baseline first to warm cache
    backtest_queue.enqueue(baseline.id)
    backtest_queue.enqueue(active.id)

    return {
        "message": f"ML A/B comparison started: baseline #{baseline.id} vs active #{active.id}",
        "baseline_id": baseline.id,
        "active_id": active.id,
        "min_confidence": min_confidence,
        "strategy": strategy,
    }


@router.post("/compare-matrix")
async def compare_ml_matrix(
    start_date: date = Query(...),
    end_date: date = Query(...),
    starting_cash: float = Query(25000.0),
    strategy: str = Query("nostate_optimized"),
    stock_universe: str = Query("all"),
    weight: float = Query(20.0, description="ML bonus weight when log_only=false"),
    min_confidence: float = Query(0.30, description="ML min_confidence for veto variants"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Launch a 4-way ML A/B/C/D matrix to separate the bonus and veto effects.

    Useful when deciding whether to graduate ML from log-only mode. The
    paired /compare endpoint conflates the two effects; this endpoint runs
    them in isolation so we can attribute any lift to the right mechanism.

    Variants (all on the same strategy + window for a fair comparison):
        A. Baseline:    log_only=true,  min_conf=0       — current production
        B. Bonus only:  log_only=false, min_conf=0       — ML modulates score
        C. Veto only:   log_only=true,  min_conf=>0      — ML can skip but not boost
        D. Both:        log_only=false, min_conf=>0      — full activation
    """
    from backend.backtest_queue import backtest_queue

    suffix = f"{strategy} {start_date}→{end_date}"

    variants = [
        ("A baseline",   {"enabled": True, "log_only": True,  "min_confidence": 0.0,             "weight": weight}),
        ("B bonus-only", {"enabled": True, "log_only": False, "min_confidence": 0.0,             "weight": weight}),
        ("C veto-only",  {"enabled": True, "log_only": True,  "min_confidence": min_confidence,  "weight": weight, "veto_action": "skip"}),
        ("D both",       {"enabled": True, "log_only": False, "min_confidence": min_confidence,  "weight": weight, "veto_action": "skip"}),
    ]

    runs = []
    for label, ml_cfg in variants:
        bt = BacktestRun(
            user_id=current_user.id,
            name=f"[ML {label}] {suffix}",
            start_date=start_date,
            end_date=end_date,
            starting_cash=starting_cash,
            stock_universe=stock_universe,
            strategy=strategy,
            status="pending",
            profile_overrides={"ml_signal": ml_cfg},
        )
        db.add(bt)
        db.flush()
        runs.append({"label": label, "id": bt.id, "ml_signal": ml_cfg})
    db.commit()

    # Enqueue baseline first so the price/score caches warm; the others reuse
    # the same data so they finish faster than the leading run.
    for r in runs:
        backtest_queue.enqueue(r["id"])

    return {
        "message": f"ML matrix started: {len(runs)} backtests queued",
        "runs": runs,
        "strategy": strategy,
        "window": f"{start_date} → {end_date}",
    }


@router.get("/cache-stats")
async def get_ml_cache_stats(
    current_user: User = Depends(get_current_active_user),
):
    """Get ML prediction cache statistics."""
    try:
        from ml.model import get_prediction_cache_stats
        return get_prediction_cache_stats()
    except ImportError:
        return {"size": 0, "hits": 0, "misses": 0}


_MATRIX_LABELS = ("A baseline", "B bonus-only", "C veto-only", "D both")
_MATRIX_PREFIX_A = "[ML A baseline] "


def _variant_payload(run: BacktestRun) -> dict:
    overrides = run.profile_overrides or {}
    if isinstance(overrides, str):
        try:
            import json as _json
            overrides = _json.loads(overrides)
        except Exception:
            overrides = {}
    return {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "return_pct": run.total_return_pct,
        "sharpe": run.sharpe_ratio,
        "max_drawdown_pct": run.max_drawdown_pct,
        "total_trades": run.total_trades,
        "win_rate": run.win_rate,
        "ml_signal": (overrides or {}).get("ml_signal") if isinstance(overrides, dict) else None,
    }


@router.get("/matrices")
async def list_ml_matrices(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List recent 4-way ML A/B/C/D matrix runs, grouped by their shared suffix.

    Detection: rows are grouped by name prefix because /compare-matrix has no
    foreign-key linking the four siblings — they only share the trailing
    "<strategy> <start>→<end>" suffix. We require all four labels to be
    present before returning a matrix; orphaned/incomplete sets are skipped.
    """
    anchors = (
        db.query(BacktestRun)
        .filter(BacktestRun.name.like(f"{_MATRIX_PREFIX_A}%"))
        .order_by(desc(BacktestRun.created_at))
        .limit(limit * 4)
        .all()
    )

    matrices = []
    for anchor in anchors:
        suffix = anchor.name[len(_MATRIX_PREFIX_A):]
        sibling_names = [f"[ML {label}] {suffix}" for label in _MATRIX_LABELS]
        siblings = (
            db.query(BacktestRun)
            .filter(BacktestRun.name.in_(sibling_names))
            .all()
        )
        by_name = {s.name: s for s in siblings}
        if not all(name in by_name for name in sibling_names):
            continue

        variants = {}
        for label in _MATRIX_LABELS:
            full_name = f"[ML {label}] {suffix}"
            variants[label] = _variant_payload(by_name[full_name])

        matrices.append({
            "strategy": anchor.strategy,
            "start_date": anchor.start_date.isoformat() if anchor.start_date else None,
            "end_date": anchor.end_date.isoformat() if anchor.end_date else None,
            "created_at": anchor.created_at.isoformat() if anchor.created_at else None,
            "suffix": suffix,
            "variants": variants,
        })

        if len(matrices) >= limit:
            break

    return {"matrices": matrices, "count": len(matrices)}
