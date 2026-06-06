import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

# for robust unpredictible stage 7 miRs
def main():
    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    # --- Параметры (можно менять) ---
    MAX_TARGETS = None  # например: 5 для теста; None = все current_targets
    RANDOM_STATE = 42

    # Boruta может быть тяжелым; для старта уменьшаем лимиты
    BORUTA_MAX_ITER = 50
    BORUTA_N_ESTIMATORS = 300
    RF_N_ESTIMATORS = 300

    X_path = Path("X_SS.parquet")
    Y_path = Path("Y_SS.parquet")
    config_path = Path("FINAL_cascade_config_v2.json")

    out_dir = Path("boruta_stage7")
    out_dir.mkdir(exist_ok=True)

    out_json = out_dir / "selected_features.json"
    out_counts_csv = out_dir / "selected_features_counts.csv"
    out_progress = out_dir / "progress.txt"

    # --- Загрузка данных ---
    log("Загружаю X_SS.parquet и Y_SS.parquet")
    X_ss = pd.read_parquet(X_path)
    y_ss = pd.read_parquet(Y_path)

    with config_path.open("r") as f:
        config = json.load(f)

    current_targets = config["stages"]["stage_7"]

    targets = list(current_targets)
    if MAX_TARGETS is not None:
        targets = targets[:MAX_TARGETS]

    feature_names = X_ss.columns.tolist()
    X = X_ss.to_numpy()

    # --- Чтобы можно было продолжать после остановки ---
    results: dict[str, list[str]] = {}
    if out_json.exists():
        results = json.loads(out_json.read_text())

    log(f"X shape = {X_ss.shape}, Y shape = {y_ss.shape}")
    log(f"Будем запускать Boruta для {len(targets)} таргетов")

    for i, tgt in enumerate(targets, start=1):
        if tgt in results:
            log(f"[{i}/{len(targets)}] {tgt}: уже есть в results, пропускаю")
            continue

        log(f"[{i}/{len(targets)}] START Boruta для таргета: {tgt}")
        out_progress.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} running {i}/{len(targets)} target={tgt}\n")
        y = y_ss[tgt].to_numpy()

        rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        boruta = BorutaPy(
            estimator=rf,
            n_estimators=BORUTA_N_ESTIMATORS,
            perc=100,
            alpha=0.05,
            two_step=True,
            max_iter=BORUTA_MAX_ITER,
            random_state=RANDOM_STATE,
            verbose=2,
        )

        t0 = time.time()
        boruta.fit(X, y)
        dt = time.time() - t0

        selected = [feature_names[idx] for idx, ok in enumerate(boruta.support_) if ok]
        results[tgt] = selected
        log(f"[{i}/{len(targets)}] DONE target={tgt} selected={len(selected)} time_s={dt:.1f}")

        # Сохраняем после каждого таргета, чтобы ничего не потерять
        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    counts = {t: len(v) for t, v in results.items()}
    pd.Series(counts).sort_values(ascending=False).to_csv(out_counts_csv)

    out_progress.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} done\n")
    log("Готово. Сохранено:")
    log(f"- {out_json}")
    log(f"- {out_counts_csv}")


if __name__ == "__main__":
    main()

