import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    # --- Inputs ---
    X_path = Path("X_SS.parquet")
    Y_path = Path("Y_SS.parquet")
    selected_path = Path("spearman_lasso_stage7/selected_features.json")

    # --- Eval settings ---
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    out_dir = Path("linear_eval_stage7")
    out_dir.mkdir(exist_ok=True)
    out_metrics = out_dir / "metrics.csv"
    out_details = out_dir / "details.json"

    log("Загружаю данные")
    X_ss = pd.read_parquet(X_path)
    y_ss = pd.read_parquet(Y_path)

    selected = json.loads(selected_path.read_text())
    targets = list(selected.keys())
    log(f"Таргеты: {targets}")

    rows = []
    details = {}

    for tgt in targets:
        feats = selected[tgt]
        if not feats:
            rows.append(
                {
                    "target": tgt,
                    "n_features": 0,
                    "r2_test": np.nan,
                    "spearman_test": np.nan,
                    "spearman_pvalue": np.nan,
                }
            )
            details[tgt] = {"features": []}
            continue

        # некоторые фичи могут отсутствовать, если менялся X
        feats_present = [f for f in feats if f in X_ss.columns]
        if not feats_present:
            rows.append(
                {
                    "target": tgt,
                    "n_features": 0,
                    "r2_test": np.nan,
                    "spearman_test": np.nan,
                    "spearman_pvalue": np.nan,
                }
            )
            details[tgt] = {"features": [], "warning": "no selected features present in X_ss"}
            continue

        X = X_ss[feats_present].to_numpy()
        y = y_ss[tgt].to_numpy()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LinearRegression(n_jobs=-1),
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        r2 = float(r2_score(y_te, y_pred))
        rho, p = spearmanr(y_te, y_pred)

        rows.append(
            {
                "target": tgt,
                "n_features": int(len(feats_present)),
                "r2_test": r2,
                "spearman_test": float(rho),
                "spearman_pvalue": float(p),
            }
        )

        details[tgt] = {
            "features": feats_present,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        }

        log(f"{tgt}: n_features={len(feats_present)} r2_test={r2:.4f} spearman={rho:.4f}")

    df = pd.DataFrame(rows).sort_values("r2_test", ascending=False, na_position="last")
    df.to_csv(out_metrics, index=False)
    out_details.write_text(json.dumps(details, ensure_ascii=False, indent=2))

    log("Готово. Сохранено:")
    log(f"- {out_metrics}")
    log(f"- {out_details}")


if __name__ == "__main__":
    main()

