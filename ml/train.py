"""Train and export a compact class-weighted fraud classifier."""
from pathlib import Path
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "creditcard.csv"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET = "Class"


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Put creditcard.csv at {DATA} (see data/README.md).")
    frame = pd.read_csv(DATA)
    missing = sorted(set(FEATURES + [TARGET]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET].astype(int), test_size=0.2,
        random_state=42, stratify=frame[TARGET].astype(int))
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)
    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(train_scaled, y_train)
    probability = model.predict_proba(test_scaled)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    metrics = {
        "rows": int(len(frame)), "fraud_rows": int(frame[TARGET].sum()),
        "fraud_rate": float(frame[TARGET].mean()), "test_rows": int(len(y_test)),
        "threshold": 0.5,
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, probability)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp),
                              "false_negative": int(fn), "true_positive": int(tp)},
    }
    artifact = {
        "model": "class_weighted_logistic_regression", "threshold": 0.5,
        "features": FEATURES, "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(), "weights": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]), "metrics": metrics,
    }
    (ROOT / "model.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Exported model to {ROOT / 'model.json'}")


if __name__ == "__main__":
    main()
