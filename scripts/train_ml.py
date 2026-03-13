#!/usr/bin/env python3
"""Offline ML model training script. Run inside Docker or locally."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

from database import SessionLocal, MLModel
from ml.feature_extractor import extract_training_data
from ml.trainer import train_model, save_model
from datetime import datetime, timezone

db = SessionLocal()

print("Extracting training data...")
df = extract_training_data(db, strategy="nostate_optimized")
print(f"Extracted {len(df)} labeled trades")
print(f"Win rate: {df['win'].mean():.1%}")
print(f"Mean gain: {df['gain_pct'].mean():.1f}%")
print()

print("Training model...")
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

if result.get("feature_importance"):
    print("\nFeature importance:")
    for feat, imp in result["feature_importance"].items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:30s} {imp:.4f} {bar}")

if result.get("baseline_comparison"):
    bl = result["baseline_comparison"]
    if bl.get("roc_auc"):
        print(f"\nBaseline LogReg AUC: {bl['roc_auc']:.4f} (vs XGBoost {result.get('mean_roc_auc', 'N/A')})")

if result.get("passed_gate") and result.get("model"):
    path = save_model(result["model"], {
        "strategy": "nostate_optimized",
        "training_samples": result.get("training_samples"),
        "feature_importance": result.get("feature_importance"),
    })
    print(f"\nModel saved to {path}")

    # Save to DB
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
        roc_auc=metrics.get("roc_auc"),
        accuracy=metrics.get("accuracy"),
        precision_score=metrics.get("precision"),
        recall_score=metrics.get("recall"),
        f1=metrics.get("f1"),
        brier_score=metrics.get("brier_score"),
        cv_results=result.get("cv_results"),
        feature_importance=result.get("feature_importance"),
        model_path=str(path),
        activated_at=datetime.now(timezone.utc),
    )
    db.add(ml_record)
    db.commit()
    print(f"Model v{version} saved to database")
else:
    print("\nModel FAILED gate or training error:")
    print(f"  {result.get('error', 'Unknown error')}")

db.close()
