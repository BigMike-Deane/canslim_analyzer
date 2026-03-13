#!/usr/bin/env python3
"""Offline ML model training script. Run inside Docker or locally.

Usage:
    python3 scripts/train_ml.py                # Train both classifier and regression
    python3 scripts/train_ml.py classifier     # Classifier only (v1 binary win/loss)
    python3 scripts/train_ml.py regression     # Regression only (v2 gain magnitude)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

from database import SessionLocal, MLModel
from ml.feature_extractor import extract_training_data
from ml.trainer import train_model, train_model_regression, save_model
from datetime import datetime, timezone

mode = sys.argv[1] if len(sys.argv) > 1 else "both"
if mode not in ("classifier", "regression", "both"):
    print(f"Usage: {sys.argv[0]} [classifier|regression|both]")
    sys.exit(1)

db = SessionLocal()

print("Extracting training data...")
df = extract_training_data(db, strategy="nostate_optimized")
print(f"Extracted {len(df)} labeled trades")
print(f"Win rate: {df['win'].mean():.1%}")
print(f"Mean gain: {df['gain_pct'].mean():.1f}%")
print()


def _train_classifier(df, db):
    """Train v1 binary classifier."""
    print("=" * 60)
    print("CLASSIFIER (v1): Binary win/loss prediction")
    print("=" * 60)
    result = train_model(df)
    print(f"Passed gate: {result['passed_gate']}")

    if result.get("mean_roc_auc"):
        print(f"Mean ROC AUC: {result['mean_roc_auc']:.4f}")

    if result.get("cv_results"):
        for fold in result["cv_results"]:
            print(f"  Fold {fold['fold']}: AUC={fold['roc_auc']:.4f}, "
                  f"Acc={fold['accuracy']:.3f}, Prec={fold.get('precision', 0):.3f}, "
                  f"Recall={fold.get('recall', 0):.3f}, F1={fold.get('f1', 0):.3f}, "
                  f"train={fold['train_size']}, test={fold['test_size']}")

    _print_importance(result)
    _print_baseline(result, "roc_auc", "LogReg")
    return result


def _train_regression(df, db):
    """Train v2 regression model."""
    print("=" * 60)
    print("REGRESSION (v2): Gain magnitude prediction")
    print("=" * 60)
    result = train_model_regression(df)
    print(f"Passed gate: {result['passed_gate']}")

    if result.get("mean_spearman"):
        print(f"Mean Spearman: {result['mean_spearman']:.4f}")

    if result.get("cv_results"):
        for fold in result["cv_results"]:
            print(f"  Fold {fold['fold']}: Spearman={fold['spearman']:.4f}, "
                  f"R²={fold['r2']:.3f}, MAE={fold['mae']:.2f}, "
                  f"DirAcc={fold.get('direction_accuracy', 0):.3f}, "
                  f"train={fold['train_size']}, test={fold['test_size']}")

    if result.get("gain_stats"):
        gs = result["gain_stats"]
        print(f"\nGain stats: mean={gs['mean']}%, median={gs['median']}%, "
              f"std={gs['std']}%, clip=[{gs['clip_range'][0]}%, {gs['clip_range'][1]}%]")

    _print_importance(result)
    _print_baseline(result, "spearman", "Ridge")
    return result


def _print_importance(result):
    if result.get("feature_importance"):
        print("\nFeature importance:")
        for feat, imp in result["feature_importance"].items():
            bar = "█" * int(imp * 50)
            print(f"  {feat:30s} {imp:.4f} {bar}")


def _print_baseline(result, metric, name):
    bl = result.get("baseline_comparison", {})
    if bl.get(metric):
        model_val = result.get(f"mean_{metric}", result.get("mean_roc_auc", "N/A"))
        print(f"\nBaseline {name} {metric}: {bl[metric]:.4f} (vs XGBoost {model_val})")


def _save_result(result, model_type, db):
    """Save model to disk and DB."""
    if not result.get("passed_gate") or not result.get("model"):
        print(f"\n{model_type.upper()} FAILED gate or training error:")
        print(f"  {result.get('error', 'Unknown error')}")
        return

    path = save_model(result["model"], {
        "strategy": "nostate_optimized",
        "model_type": model_type,
        "training_samples": result.get("training_samples"),
        "feature_importance": result.get("feature_importance"),
    })
    print(f"\nModel saved to {path}")

    latest = db.query(MLModel).filter(
        MLModel.strategy == "nostate_optimized"
    ).order_by(MLModel.version.desc()).first()
    version = (latest.version + 1) if latest else 1

    metrics = result.get("metrics", {})
    ml_record = MLModel(
        version=version,
        strategy="nostate_optimized",
        status="active",
        training_samples=result.get("training_samples"),
        feature_count=result.get("feature_count"),
        roc_auc=metrics.get("roc_auc") or metrics.get("spearman"),
        accuracy=metrics.get("accuracy") or metrics.get("direction_accuracy"),
        precision_score=metrics.get("precision"),
        recall_score=metrics.get("recall"),
        f1=metrics.get("f1"),
        brier_score=metrics.get("brier_score") or metrics.get("mae"),
        cv_results=result.get("cv_results"),
        feature_importance=result.get("feature_importance"),
        model_path=str(path),
        activated_at=datetime.now(timezone.utc),
    )
    db.add(ml_record)
    db.commit()
    print(f"Model v{version} ({model_type}) saved to database")


# Run training
if mode in ("classifier", "both"):
    cls_result = _train_classifier(df, db)
    if mode == "classifier":
        _save_result(cls_result, "classifier", db)

if mode in ("regression", "both"):
    if mode == "both":
        print("\n")
    reg_result = _train_regression(df, db)

# When training both, save the BETTER model
if mode == "both":
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    cls_passed = cls_result.get("passed_gate", False)
    reg_passed = reg_result.get("passed_gate", False)

    if cls_passed:
        print(f"  Classifier: ROC AUC = {cls_result.get('mean_roc_auc', 0):.4f} ✓")
    else:
        print(f"  Classifier: FAILED gate")
    if reg_passed:
        print(f"  Regression: Spearman = {reg_result.get('mean_spearman', 0):.4f} ✓")
    else:
        print(f"  Regression: FAILED gate")

    # Save regression if it passes (preferred for gain-aware ranking)
    if reg_passed:
        print("\n→ Saving REGRESSION model (gain-aware ranking)")
        _save_result(reg_result, "regression", db)
    elif cls_passed:
        print("\n→ Saving CLASSIFIER model (regression failed gate)")
        _save_result(cls_result, "classifier", db)
    else:
        print("\n→ No model saved (both failed gate)")

if mode == "regression":
    _save_result(reg_result, "regression", db)

db.close()
