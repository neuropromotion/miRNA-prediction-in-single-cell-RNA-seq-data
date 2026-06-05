import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LOG_PATH: Path | None = None


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_PATH is None:
        return
    # Пишем в файл лога, если доступно. Если нет — не ломаем пайплайн.
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def append_progress(path: Path, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a") as f:
        f.write(f"{ts} {msg}\n")


def spearman_filter_indices(X: np.ndarray, y: np.ndarray, thr: float, chunk: int) -> np.ndarray:
    y_rank = rankdata(y, method="average").astype(np.float32)
    y_rank = (y_rank - y_rank.mean()) / (y_rank.std() + 1e-12)

    selected: list[int] = []
    n_features = X.shape[1]

    for start in range(0, n_features, chunk):
        end = min(start + chunk, n_features)
        Xc = X[:, start:end]

        # spearman = pearson(rank(X), rank(y))
        Xr = np.apply_along_axis(rankdata, 0, Xc, method="average").astype(np.float32)
        Xr = Xr - Xr.mean(axis=0)
        Xr = Xr / (Xr.std(axis=0) + 1e-12)

        rho = (Xr * y_rank[:, None]).mean(axis=0)
        keep = np.where(np.abs(rho) >= thr)[0]
        if keep.size:
            selected.extend((start + keep).tolist())

    return np.array(selected, dtype=int)


def main():
    # --- Настройки ---
    SPEARMAN_THR_HIGH = 0.2
    SPEARMAN_THR_LOW = 0.1
    MIN_SPEARMAN_FEATURES = 100
    CHUNK = 512

    LASSO_CV = 5
    LASSO_ALPHAS = 60
    LASSO_MAX_ITER = 20000
    RANDOM_STATE = 42
    N_JOBS = -1

    X_path = Path("X_SS.parquet")
    Y_path = Path("Y_SS.parquet")
    targets_path = Path("stage_7_targets.txt")

    out_dir_name = os.environ.get("SPEARMAN_LASSO_OUT_DIR", "spearman_lasso_all_targets")
    out_dir = Path(out_dir_name)
    out_dir.mkdir(exist_ok=True)

    global LOG_PATH
    LOG_PATH = out_dir / "run.log"

    out_json = out_dir / "selected_features.json"
    out_meta = out_dir / "run_meta.json"
    out_counts_csv = out_dir / "selected_features_counts.csv"
    out_progress = out_dir / "progress.txt"
    out_details = out_dir / "details.json"
    out_failed = out_dir / "failed_targets.json"

    log("Загружаю X_SS.parquet и Y_SS.parquet")
    X_ss = pd.read_parquet(X_path)
    y_ss = pd.read_parquet(Y_path)
    log(f"X shape = {X_ss.shape}, Y shape = {y_ss.shape}")

    targets = [t.strip() for t in targets_path.read_text().splitlines() if t.strip()]
    targets = [t for t in targets if t in y_ss.columns]
    X = X_ss.to_numpy()

    results: dict[str, list[str]] = {}
    details: dict[str, dict] = {}
    failed: dict[str, str] = {}

    if out_json.exists():
        results = json.loads(out_json.read_text())
    if out_details.exists():
        details = json.loads(out_details.read_text())
    if out_failed.exists():
        failed = json.loads(out_failed.read_text())

    meta = {
        "targets_count": len(targets),
        "spearman_threshold_high": SPEARMAN_THR_HIGH,
        "spearman_threshold_low": SPEARMAN_THR_LOW,
        "min_spearman_features": MIN_SPEARMAN_FEATURES,
        "chunk": CHUNK,
        "lasso": {
            "cv": LASSO_CV,
            "alphas": LASSO_ALPHAS,
            "max_iter": LASSO_MAX_ITER,
            "random_state": RANDOM_STATE,
        },
        "X_shape": list(X_ss.shape),
        "Y_shape": list(y_ss.shape),
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    log(f"Старт пайплайна для всех таргетов: {len(targets)}")

    for i, tgt in enumerate(targets, start=1):
        if tgt in results:
            log(f"[{i}/{len(targets)}] {tgt}: уже есть результат, пропускаю")
            continue
        if tgt in failed:
            log(f"[{i}/{len(targets)}] {tgt}: ранее failed, пропускаю")
            continue

        append_progress(out_progress, f"running {i}/{len(targets)} target={tgt}")
        log(f"[{i}/{len(targets)}] target={tgt}: Spearman фильтр")

        try:
            y = y_ss[tgt].to_numpy()

            t0 = time.time()
            idx_high = spearman_filter_indices(X=X, y=y, thr=SPEARMAN_THR_HIGH, chunk=CHUNK)
            dt_spear_high = time.time() - t0

            log(
                f"[{i}/{len(targets)}] target={tgt}: Spearman(high={SPEARMAN_THR_HIGH}) left={len(idx_high)} "
                f"(time_s={dt_spear_high:.1f})"
            )

            # Если слишком мало фич — повторяем на более мягком пороге
            idx = idx_high
            spearman_thr_used = SPEARMAN_THR_HIGH
            dt_spear_low = 0.0
            spearman_left_low = None
            if len(idx_high) < MIN_SPEARMAN_FEATURES:
                t1 = time.time()
                idx_low = spearman_filter_indices(X=X, y=y, thr=SPEARMAN_THR_LOW, chunk=CHUNK)
                dt_spear_low = time.time() - t1
                idx = idx_low
                spearman_thr_used = SPEARMAN_THR_LOW
                spearman_left_low = len(idx_low)
                log(
                    f"[{i}/{len(targets)}] target={tgt}: Spearman(low={SPEARMAN_THR_LOW}) left={len(idx_low)} "
                    f"(time_s={dt_spear_low:.1f})"
                )

            append_progress(
                out_progress,
                f"target={tgt} spearman_thr_used={spearman_thr_used} "
                f"spearman_left_high={len(idx_high)} spearman_left_final={len(idx)} "
                f"time_s={(dt_spear_high + dt_spear_low):.1f}s",
            )

            if len(idx) == 0:
                results[tgt] = []
                details[tgt] = {
                    "n_spearman": 0,
                    "n_lasso_nonzero": 0,
                    "spearman_thr_used": spearman_thr_used,
                    "spearman_left_high": int(len(idx_high)),
                    "spearman_left_low": None if spearman_left_low is None else int(spearman_left_low),
                    "spearman_time_s": float(dt_spear_high + dt_spear_low),
                    "lasso_time_s": 0.0,
                }
            else:
                X_f = X_ss.iloc[:, idx]

                log(f"[{i}/{len(targets)}] target={tgt}: LassoCV на {X_f.shape[1]} фичах")
                model = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    LassoCV(
                        cv=LASSO_CV,
                        alphas=LASSO_ALPHAS,
                        max_iter=LASSO_MAX_ITER,
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS,
                    ),
                )

                t1 = time.time()
                model.fit(X_f, y)
                dt_lasso = time.time() - t1

                coef = model.named_steps["lassocv"].coef_
                nonzero = np.where(coef != 0)[0]
                selected = X_f.columns[nonzero].tolist()

                results[tgt] = selected
                details[tgt] = {
                    "n_spearman": int(X_f.shape[1]),
                    "spearman_thr_used": spearman_thr_used,
                    "n_lasso_nonzero": int(len(selected)),
                    "spearman_time_s": float(dt_spear_high + dt_spear_low),
                    "lasso_time_s": float(dt_lasso),
                }
                log(
                    f"[{i}/{len(targets)}] target={tgt}: Lasso оставил {len(selected)} "
                    f"(time_s={dt_lasso:.1f})"
                )
                append_progress(
                    out_progress,
                    f"target={tgt} lasso_nonzero={len(selected)} time_lasso_s={dt_lasso:.1f}s",
                )

        except Exception as e:
            failed[tgt] = repr(e)
            log(f"[{i}/{len(targets)}] target={tgt}: ERROR {e!r}")
            append_progress(out_progress, f"target={tgt} ERROR {repr(e)}")

        # сохраняем после каждого таргета
        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        out_details.write_text(json.dumps(details, ensure_ascii=False, indent=2))
        out_failed.write_text(json.dumps(failed, ensure_ascii=False, indent=2))

    counts = {t: len(v) for t, v in results.items()}
    pd.Series(counts).sort_values(ascending=False).to_csv(out_counts_csv)
    append_progress(out_progress, "done")

    log("Готово.")
    log(f"Успешно: {len(results)} таргетов")
    log(f"Ошибок: {len(failed)} таргетов")
    log(f"Артефакты в {out_dir}")


if __name__ == "__main__":
    main()

