"""ML Signal Layer API routes."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database import (
    get_db, BacktestRun, BacktestTrade, MLModel, MLPrediction, User,
)
from backend.auth import get_current_active_user, get_admin_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ml"])


def _run_training(db_url: str, strategy: str, backtest_ids: list, ml_model_id: int, mode: str = "regression"):
    """Background task: extract features, train model, update DB record.

    mode: 'classifier', 'regression', or 'both' (both trains both, saves regression if it passes).
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
        from ml.feature_extractor import extract_training_data
        from ml.trainer import train_model, train_model_regression, save_model

        ml_record = db.query(MLModel).get(ml_model_id)
        if not ml_record:
            return

        # Extract training data
        df = extract_training_data(db, strategy=strategy, backtest_ids=backtest_ids or None)
        if df.empty:
            ml_record.status = "failed"
            ml_record.error_message = "No training data extracted"
            db.commit()
            return

        ml_record.training_samples = len(df)
        db.commit()

        # Train based on mode
        result = None
        if mode == "classifier":
            result = train_model(df)
        elif mode == "regression":
            result = train_model_regression(df)
        elif mode == "both":
            # Train both, prefer regression if it passes
            cls_result = train_model(df)
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

        # Save model to disk
        model = result["model"]
        metrics = result["metrics"]
        model_path = save_model(model, {
            "strategy": strategy,
            "version": ml_record.version,
            "training_samples": result["training_samples"],
            "feature_importance": result["feature_importance"],
            "model_type": model_type,
        })

        # Update DB record
        ml_record.status = "completed"
        ml_record.model_type = model_type

        if model_type == "regression":
            ml_record.spearman = metrics.get("spearman")
            ml_record.r2_score = metrics.get("r2")
            ml_record.mae = metrics.get("mae")
            ml_record.direction_accuracy = metrics.get("direction_accuracy")
        else:
            ml_record.roc_auc = metrics.get("roc_auc")
            ml_record.accuracy = metrics.get("accuracy")
            ml_record.precision_score = metrics.get("precision")
            ml_record.recall_score = metrics.get("recall")
            ml_record.f1 = metrics.get("f1")
            ml_record.brier_score = metrics.get("brier_score")

        ml_record.cv_results = result.get("cv_results")
        ml_record.feature_importance = result.get("feature_importance")
        ml_record.feature_count = result.get("feature_count")
        ml_record.model_path = str(model_path)
        ml_record.training_samples = result.get("training_samples")

        # Deactivate previous active model for this strategy
        db.query(MLModel).filter(
            MLModel.strategy == strategy,
            MLModel.status == "active",
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

        primary_metric = f"Spearman={ml_record.spearman}" if model_type == "regression" else f"ROC AUC={ml_record.roc_auc}"
        logger.info(f"ML model v{ml_record.version} ({model_type}) trained and activated: {primary_metric}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        try:
            ml_record = db.query(MLModel).get(ml_model_id)
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
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Trigger ML model training (admin only). Runs in background."""
    if mode not in ("classifier", "regression", "both"):
        raise HTTPException(400, "mode must be 'classifier', 'regression', or 'both'")

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
    background_tasks.add_task(_run_training, DATABASE_URL, strategy, ids, ml_record.id, mode)

    return {
        "message": f"Training started for v{next_version} ({mode})",
        "model_id": ml_record.id,
        "version": next_version,
        "strategy": strategy,
        "mode": mode,
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
    """Preview extracted training dataset (admin only)."""
    try:
        from ml.feature_extractor import extract_training_data
        df = extract_training_data(db, strategy=strategy)
    except Exception as e:
        raise HTTPException(500, f"Feature extraction failed: {e}")

    if df.empty:
        return {"total": 0, "win_rate": None, "samples": []}

    return {
        "total": len(df),
        "win_rate": round(float(df["win"].mean()), 4),
        "mean_gain": round(float(df["gain_pct"].mean()), 2),
        "samples": df.head(limit).to_dict(orient="records"),
    }
