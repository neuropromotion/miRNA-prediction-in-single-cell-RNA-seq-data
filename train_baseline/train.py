import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle


SEED = 42

# Split config
BULK_TEST_SIZE = 0.15
BULK_VAL_SIZE = 0.20
SINGLE_CELL_VAL_SIZE = 0.20

# Optuna / XGBoost config
N_TRIALS = int(os.environ.get("CASCADE_N_TRIALS", "5"))
N_JOBS = int(os.environ.get("CASCADE_N_JOBS", "-1"))
OOF_N_SPLITS = int(os.environ.get("CASCADE_OOF_SPLITS", "3"))
STAGE_TUNING_TARGETS = int(os.environ.get("CASCADE_STAGE_TUNING_TARGETS", "3"))
XGB_DEVICE = os.environ.get("CASCADE_XGB_DEVICE", "cuda")
XGB_TREE_METHOD = os.environ.get("CASCADE_XGB_TREE_METHOD", "hist")

OUTPUT_DIR = Path(os.environ.get("CASCADE_OUTPUT_DIR", "cascade_training_v3"))
LOG_PATH = OUTPUT_DIR / "run.log"


def log(message: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sort_stage_names(stages: dict) -> list[str]:
    return sorted(stages.keys(), key=lambda x: int(x.split("_")[-1]))


def metrics_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    r2 = float(r2_score(y_true, y_pred))
    rho, pvalue = spearmanr(y_true, y_pred)
    rho = float(rho) if not math.isnan(rho) else -1.0
    pvalue = float(pvalue) if not math.isnan(pvalue) else 1.0
    return {
        "r2": r2,
        "spearman": rho,
        "spearman_pvalue": pvalue,
        "score": 0.5 * (r2 + rho),
    }


def build_feature_frame(
    feature_list: list[str],
    base_features: pd.DataFrame,
    cascade_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    parts = []
    mrna_used = []
    cascade_used = []
    missing = []

    for feature_name in feature_list:
        if feature_name in base_features.columns:
            parts.append(base_features[[feature_name]])
            mrna_used.append(feature_name)
        elif feature_name in cascade_features.columns:
            parts.append(cascade_features[[feature_name]])
            cascade_used.append(feature_name)
        else:
            missing.append(feature_name)

    if not parts:
        return pd.DataFrame(index=base_features.index), mrna_used, cascade_used, missing

    return pd.concat(parts, axis=1), mrna_used, cascade_used, missing


def make_xgb_regressor(params: dict) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method=XGB_TREE_METHOD,
        device=XGB_DEVICE,
        random_state=SEED,
        n_jobs=N_JOBS,
        **params,
    )


def suggest_xgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 3.0, log=True),
    }


def make_kfold(n_samples: int) -> KFold:
    n_splits = min(OOF_N_SPLITS, n_samples)
    if n_splits < 2:
        raise ValueError("Not enough samples for OOF scheme")
    return KFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def cross_val_score_target(X: pd.DataFrame, y: pd.Series, params: dict) -> float:
    kf = make_kfold(len(X))
    scores = []

    for train_idx, valid_idx in kf.split(X):
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]
        y_fold_valid = y.iloc[valid_idx]

        model = make_xgb_regressor(params)
        model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_valid, y_fold_valid)], verbose=False)
        pred_valid = model.predict(X_fold_valid)
        scores.append(metrics_dict(y_fold_valid, pred_valid)["score"])

    return float(np.mean(scores))


def fit_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kf = make_kfold(len(X_train))

    oof_pred = np.zeros(len(X_train), dtype=float)
    val_pred_folds = []
    test_pred_folds = []

    for train_idx, valid_idx in kf.split(X_train):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        model = make_xgb_regressor(params)
        model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_valid, y_fold_valid)], verbose=False)

        oof_pred[valid_idx] = model.predict(X_fold_valid)
        val_pred_folds.append(model.predict(X_val))
        test_pred_folds.append(model.predict(X_test))

    val_pred = np.mean(np.vstack(val_pred_folds), axis=0)
    test_pred = np.mean(np.vstack(test_pred_folds), axis=0)
    return oof_pred, val_pred, test_pred


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged_rna = np.log1p(pd.read_parquet("merged_rna.parquet"))
    merged_mir = np.log1p(pd.read_parquet("merged_mir.parquet"))

    X_ss = pd.read_parquet("X_SS.parquet")
    Y_ss = pd.read_parquet("Y_SS.parquet")

    X_ss_pure = pd.read_parquet("X_ss_pure.parquet").T
    Y_ss_pure = pd.read_parquet("y_ss_pure.parquet").T

    assert merged_rna.columns.equals(X_ss.columns)
    assert merged_rna.columns.equals(X_ss_pure.columns)
    assert merged_mir.columns.equals(Y_ss.columns)
    assert merged_mir.columns.equals(Y_ss_pure.columns)

    assert merged_rna.index.equals(merged_mir.index)
    assert X_ss.index.equals(Y_ss.index)
    assert X_ss_pure.index.equals(Y_ss_pure.index)

    X_ss_full = pd.concat([X_ss, X_ss_pure], axis=0)
    Y_ss_full = pd.concat([Y_ss, Y_ss_pure], axis=0)

    assert X_ss_full.index.equals(Y_ss_full.index)
    assert merged_rna.columns.equals(X_ss_full.columns)
    assert merged_mir.columns.equals(Y_ss_full.columns)

    return merged_rna, merged_mir, X_ss_full, Y_ss_full


def make_splits(
    merged_rna: pd.DataFrame,
    merged_mir: pd.DataFrame,
    X_ss_full: pd.DataFrame,
    Y_ss_full: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    bulk_remaining_idx, bulk_test_idx = train_test_split(
        merged_rna.index,
        test_size=BULK_TEST_SIZE,
        random_state=SEED,
    )

    X_test_bulk = merged_rna.loc[bulk_test_idx]
    y_test_bulk = merged_mir.loc[bulk_test_idx]

    bulk_train_idx, bulk_val_idx = train_test_split(
        bulk_remaining_idx,
        test_size=BULK_VAL_SIZE,
        random_state=SEED,
    )

    ss_train_idx, ss_val_idx = train_test_split(
        X_ss_full.index,
        test_size=SINGLE_CELL_VAL_SIZE,
        random_state=SEED,
    )

    X_train = pd.concat([merged_rna.loc[bulk_train_idx], X_ss_full.loc[ss_train_idx]], axis=0)
    y_train = pd.concat([merged_mir.loc[bulk_train_idx], Y_ss_full.loc[ss_train_idx]], axis=0)
    X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

    X_val = pd.concat([merged_rna.loc[bulk_val_idx], X_ss_full.loc[ss_val_idx]], axis=0)
    y_val = pd.concat([merged_mir.loc[bulk_val_idx], Y_ss_full.loc[ss_val_idx]], axis=0)

    assert X_train.index.equals(y_train.index)
    assert X_val.index.equals(y_val.index)
    assert X_test_bulk.index.equals(y_test_bulk.index)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test_bulk": X_test_bulk,
        "y_test_bulk": y_test_bulk,
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def pick_stage_tuning_targets(stage_targets: list[str], feature_map: dict) -> list[str]:
    available = [t for t in stage_targets if t in feature_map]
    return available[: min(STAGE_TUNING_TARGETS, len(available))]


def tune_stage_params(
    stage_name: str,
    stage_targets: list[str],
    feature_map: dict,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    train_pred_pool: pd.DataFrame,
) -> tuple[dict, list[str]]:
    tuning_targets = pick_stage_tuning_targets(stage_targets, feature_map)
    if not tuning_targets:
        raise ValueError(f"No tunable targets found for {stage_name}")

    log(f"[{stage_name}] tuning one Optuna study for stage on targets={tuning_targets}")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial)
        stage_scores = []

        for target in tuning_targets:
            feature_list = feature_map[target]
            X_target, _, _, _ = build_feature_frame(feature_list, X_train, train_pred_pool)
            if X_target.shape[1] == 0:
                continue
            y_target = y_train[target]
            stage_scores.append(cross_val_score_target(X_target, y_target, params))

        if not stage_scores:
            return -1.0
        return float(np.mean(stage_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])

    save_json(
        OUTPUT_DIR / "params" / f"{stage_name}_best_params.json",
        {
            "stage": stage_name,
            "tuning_targets": tuning_targets,
            "best_value": float(study.best_value),
            "best_params": best_params,
        },
    )

    log(f"[{stage_name}] stage-level params selected | best_score={study.best_value:.4f}")
    return best_params, tuning_targets


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(exist_ok=True)
    (OUTPUT_DIR / "params").mkdir(exist_ok=True)

    log("Loading datasets")
    merged_rna, merged_mir, X_ss_full, Y_ss_full = load_data()

    log("Loading cascade config")
    with open("FINAL_cascade_config_v3.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    stages = config["stages"]
    feature_map = config["features"]
    ordered_stages = sort_stage_names(stages)

    log("Preparing train/val/test split")
    splits = make_splits(merged_rna, merged_mir, X_ss_full, Y_ss_full)
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test_bulk = splits["X_test_bulk"]
    y_test_bulk = splits["y_test_bulk"]

    log(f"TRAIN: {X_train.shape}")
    log(f"VAL: {X_val.shape}")
    log(f"TEST_BULK: {X_test_bulk.shape}")
    log(f"Using strict OOF cascade with {min(OOF_N_SPLITS, len(X_train))} folds")
    log(
        f"Fast mode: one Optuna study per stage | "
        f"device={XGB_DEVICE} tree_method={XGB_TREE_METHOD} "
        f"n_trials={N_TRIALS} tuning_targets_per_stage={STAGE_TUNING_TARGETS}"
    )

    train_pred_pool = pd.DataFrame(index=X_train.index)
    val_pred_pool = pd.DataFrame(index=X_val.index)
    test_pred_pool = pd.DataFrame(index=X_test_bulk.index)

    results = []
    failed_targets = {}

    for stage_name in ordered_stages:
        stage_targets = stages[stage_name]
        log(f"=== {stage_name} | targets={len(stage_targets)} ===")

        try:
            stage_params, tuning_targets = tune_stage_params(
                stage_name=stage_name,
                stage_targets=stage_targets,
                feature_map=feature_map,
                X_train=X_train,
                y_train=y_train,
                train_pred_pool=train_pred_pool,
            )
        except Exception as exc:
            for target in stage_targets:
                failed_targets[target] = {"stage": stage_name, "error": repr(exc)}
            save_json(OUTPUT_DIR / "failed_targets.json", failed_targets)
            log(f"[{stage_name}] stage-level tuning ERROR: {exc!r}")
            continue

        for pos, target in enumerate(stage_targets, start=1):
            log(f"[{stage_name} {pos}/{len(stage_targets)}] target={target} | building features")

            try:
                if target not in feature_map:
                    raise KeyError(f"No features found in config for target {target}")
                if target not in y_train.columns:
                    raise KeyError(f"Target {target} is absent in training labels")

                target_feature_list = feature_map[target]

                X_train_target, mrna_used, cascade_used, missing_features = build_feature_frame(
                    target_feature_list,
                    X_train,
                    train_pred_pool,
                )
                X_val_target, _, _, _ = build_feature_frame(target_feature_list, X_val, val_pred_pool)
                X_test_target, _, _, _ = build_feature_frame(target_feature_list, X_test_bulk, test_pred_pool)

                if X_train_target.shape[1] == 0:
                    raise ValueError("No usable features were found for this target")

                y_train_target = y_train[target]
                y_val_target = y_val[target]
                y_test_target = y_test_bulk[target]

                log(
                    f"[{stage_name}] target={target} | n_features={X_train_target.shape[1]} "
                    f"(mrna={len(mrna_used)}, cascade={len(cascade_used)}, missing={len(missing_features)})"
                )

                pred_train_oof, pred_val_stack, pred_test_stack = fit_oof_predictions(
                    X_train_target,
                    y_train_target,
                    X_val_target,
                    X_test_target,
                    stage_params,
                )

                train_pred_pool[target] = pred_train_oof
                val_pred_pool[target] = pred_val_stack
                test_pred_pool[target] = pred_test_stack

                train_metrics = metrics_dict(y_train_target, pred_train_oof)
                val_metrics = metrics_dict(y_val_target, pred_val_stack)
                test_metrics = metrics_dict(y_test_target, pred_test_stack)

                final_model = make_xgb_regressor(stage_params)
                final_model.fit(X_train_target, y_train_target, eval_set=[(X_val_target, y_val_target)], verbose=False)

                model_path = OUTPUT_DIR / "models" / stage_name / f"{target}.json"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                final_model.save_model(model_path)

                save_json(
                    OUTPUT_DIR / "params" / stage_name / f"{target}.json",
                    {
                        "target": target,
                        "stage": stage_name,
                        "stage_tuning_targets": tuning_targets,
                        "stage_params": stage_params,
                        "n_features": int(X_train_target.shape[1]),
                        "mrna_features": mrna_used,
                        "cascade_features": cascade_used,
                        "missing_features": missing_features,
                        "oof_splits": min(OOF_N_SPLITS, len(X_train_target)),
                    },
                )

                result_row = {
                    "stage": stage_name,
                    "target": target,
                    "n_features": int(X_train_target.shape[1]),
                    "n_mrna_features": int(len(mrna_used)),
                    "n_cascade_features": int(len(cascade_used)),
                    "n_missing_features": int(len(missing_features)),
                    "train_oof_r2": train_metrics["r2"],
                    "train_oof_spearman": train_metrics["spearman"],
                    "val_r2": val_metrics["r2"],
                    "val_spearman": val_metrics["spearman"],
                    "test_bulk_r2": test_metrics["r2"],
                    "test_bulk_spearman": test_metrics["spearman"],
                    "model_path": str(model_path),
                }
                results.append(result_row)
                pd.DataFrame(results).to_csv(OUTPUT_DIR / "metrics.csv", index=False)

                log(
                    f"[{stage_name}] target={target} | "
                    f"train_oof_r2={train_metrics['r2']:.4f} "
                    f"train_oof_spearman={train_metrics['spearman']:.4f} | "
                    f"val_r2={val_metrics['r2']:.4f} "
                    f"val_spearman={val_metrics['spearman']:.4f} | "
                    f"test_bulk_r2={test_metrics['r2']:.4f} "
                    f"test_bulk_spearman={test_metrics['spearman']:.4f}"
                )

            except Exception as exc:
                failed_targets[target] = {
                    "stage": stage_name,
                    "error": repr(exc),
                }
                save_json(OUTPUT_DIR / "failed_targets.json", failed_targets)
                log(f"[{stage_name}] target={target} | ERROR: {exc!r}")

    save_json(OUTPUT_DIR / "failed_targets.json", failed_targets)
    save_json(
        OUTPUT_DIR / "run_summary.json",
        {
            "n_completed": len(results),
            "n_failed": len(failed_targets),
            "stages": ordered_stages,
            "oof_splits": OOF_N_SPLITS,
            "n_trials": N_TRIALS,
            "stage_tuning_targets": STAGE_TUNING_TARGETS,
        },
    )
    log("Training finished")


if __name__ == "__main__":
    main()
