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


def _run_training(db_url: str, strategy: str, backtest_ids: list, ml_model_id: int,
                  mode: str = "regression",
                  excluded_features: Optional[list] = None,
                  calibrate: bool = False,
                  auto_activate: bool = True):
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
            result = train_model(df, excluded_features=excluded_features, calibrate=calibrate)
        elif mode == "regression":
            result = train_model_regression(df)
        elif mode == "both":
            # Train both, prefer regression if it passes
            cls_result = train_model(df, excluded_features=excluded_features, calibrate=calibrate)
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
        feat_cols = result.get("feature_count_columns") or None  # forward-compat
        active_features = list(result.get("feature_importance", {}).keys()) or None
        if is_experimental:
            from ml.trainer import MODEL_DIR
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
            model_path = save_model(model, {
                "strategy": strategy,
                "version": ml_record.version,
                "training_samples": result["training_samples"],
                "feature_importance": result["feature_importance"],
                "model_type": model_type,
                "calibrated": bool(calibrate),
            }, feature_columns=active_features)

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

        # Improvement gate: only activate if better than current active model
        current_metric, current_version = _get_active_model_metric(db, strategy, model_type)

        if current_metric is not None and new_metric is not None and new_metric < current_metric:
            # New model is worse — save as completed but don't activate
            ml_record.status = "completed"
            ml_record.error_message = (
                f"Not activated: {metric_name} {new_metric:.4f} < "
                f"current active v{current_version} {current_metric:.4f}"
            )
            db.commit()
            logger.warning(
                f"ML model v{ml_record.version} passed gate but worse than active: "
                f"{metric_name} {new_metric:.4f} < {current_metric:.4f} (v{current_version})"
            )
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

        # Reload model in memory
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
        },
    )
    db.add(ml_record)
    db.commit()
    db.refresh(ml_record)

    # Get DB URL for background task (needs its own session)
    from backend.database import DATABASE_URL
    background_tasks.add_task(
        _run_training, DATABASE_URL, strategy, ids, ml_record.id, mode,
        excluded_list or None, calibrate, auto_activate,
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
