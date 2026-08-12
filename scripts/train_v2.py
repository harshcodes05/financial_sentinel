import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, f1_score
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_V2_DIR = PROJECT_ROOT / "models" / "v2"

MODEL_V2_DIR.mkdir(parents=True, exist_ok=True)

def train_dual_models_v2():
    print("[1/5] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    scale_pos_weight = num_neg / num_pos
    print(f"Dataset imbalance ratio (Legit/Fraud): {scale_pos_weight:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[2/5] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # 1. Supervised Model (XGBoost)
    print("[3/5] Training Supervised XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train_scaled, y_train)

    # 2. Unsupervised Model (Isolation Forest Anomaly Detector)
    print("[4/5] Training Unsupervised Isolation Forest Anomaly Detector...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.005,
        random_state=42,
        n_jobs=-1
    )
    # Use .values to avoid pandas IndexingError
    iso_forest.fit(X_train_scaled[y_train.values == 0])

    # Evaluate XGBoost Threshold
    y_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
    best_idx = np.argmax(f2_scores)
    best_threshold = float(thresholds[best_idx])
    
    y_preds_opt = (y_probs >= best_threshold).astype(int)
    roc_auc = float(roc_auc_score(y_test, y_probs))
    f1 = float(f1_score(y_test, y_preds_opt))

    print(f"Optimal Decision Threshold: {best_threshold:.4f}")
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print(f"F1 Score      : {f1:.4f}")

    # Save Dual Model Artifacts
    print(f"[5/5] Saving dual model artifacts to {MODEL_V2_DIR}...")
    joblib.dump(xgb_model, MODEL_V2_DIR / "xgboost_model.pkl")
    joblib.dump(iso_forest, MODEL_V2_DIR / "isolation_forest_model.pkl")
    joblib.dump(scaler, MODEL_V2_DIR / "standard_scaler.pkl")

    metadata = {
        "models": ["XGBoostClassifier", "IsolationForest"],
        "version": "v2.1",
        "optimal_threshold": best_threshold,
        "metrics": {"roc_auc": roc_auc, "f1_score": f1},
        "feature_names": list(X.columns)
    }

    with open(MODEL_V2_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Dual-model training complete!")

if __name__ == "__main__":
    train_dual_models_v2()
