import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_V2_DIR = PROJECT_ROOT / "models" / "v2"

MODEL_V2_DIR.mkdir(parents=True, exist_ok=True)

def generate_evaluation_artifacts(y_test, y_probs, y_preds_opt, best_threshold, precisions, recalls):
    """Generates and saves Confusion Matrix and Precision-Recall Curve plots for model evaluation reproducibility."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_preds_opt)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legitimate", "Fraudulent"],
                yticklabels=["Legitimate", "Fraudulent"])
    ax.set_title(f"XGBoost v2 Confusion Matrix (Threshold = {best_threshold:.4f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    cm_path = MODEL_V2_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix artifact to {cm_path}")

    # 2. Plot Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls, precisions, color="#2563EB", linewidth=2, label="XGBoost v2 PR Curve")
    ax.set_title("Precision-Recall Curve (F2 Optimization)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left")
    plt.tight_layout()
    pr_path = MODEL_V2_DIR / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"Saved Precision-Recall Curve artifact to {pr_path}")

def train_dual_models_v2():
    print("[1/6] Loading dataset...")
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

    print("[2/6] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # 1. Supervised Model (XGBoost)
    print("[3/6] Training Supervised XGBoost Classifier...")
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
    print("[4/6] Training Unsupervised Isolation Forest Anomaly Detector...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.005,
        random_state=42,
        n_jobs=-1
    )
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

    # Generate Evaluation Artifacts
    print("[5/6] Generating evaluation visual artifacts (Confusion Matrix & PR Curve)...")
    generate_evaluation_artifacts(y_test, y_probs, y_preds_opt, best_threshold, precisions, recalls)

    # Save Dual Model Artifacts
    print(f"[6/6] Saving dual model artifacts to {MODEL_V2_DIR}...")
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

    print("Dual-model training & artifact generation complete!")

if __name__ == "__main__":
    train_dual_models_v2()
