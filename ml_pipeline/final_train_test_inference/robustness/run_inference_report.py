#!/usr/bin/env python3
"""Optional h5ad panel + rebuild robustness_config with inference_datasets blocks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

_ML = _THIS.parent.parent
_TABPACK = os.environ.get("TABPACK_ROOT", str(_ML / "deps" / "tabpack"))
_tab_src = str(Path(_TABPACK).resolve() / "src")
if _tab_src not in sys.path:
    sys.path.insert(0, _tab_src)

import robustness_paths as C  # noqa: E402
from config_builder import build_robustness_config, load_bootstrap_references  # noqa: E402
from inference_panel import assess_dataset  # noqa: E402

import common as cm  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad-healthy", type=Path, default=None)
    parser.add_argument("--h5ad-rcc", type=Path, default=None)
    parser.add_argument(
        "--no-rebuild-config",
        action="store_true",
        help="only write inference_datasets.csv",
    )
    args = parser.parse_args()

    try:
        import anndata  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "anndata is required. Install: pip install -r requirements.txt"
        ) from exc

    cal_path = C.TABLES_DIR / "calibration_summary.csv"
    if not cal_path.is_file():
        raise SystemExit(f"missing {cal_path} — run run_robustness.py first")

    cal = pd.read_csv(cal_path)
    cal_ok = cal.loc[cal["status"] == "ok"].copy()
    if cal_ok.empty:
        raise SystemExit("no successful miRNAs in calibration_summary")

    config = cm.load_target_config()
    target_info = config["targets"]
    sc_features = cm.load_sc_features()

    h5ad_map = {
        "healthy_integrated_L": args.h5ad_healthy or C.DEFAULT_INFERENCE_H5AD["healthy_integrated_L"],
        "RCC_integrated_full": args.h5ad_rcc or C.DEFAULT_INFERENCE_H5AD["RCC_integrated_full"],
    }

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] inference panel n_mir={len(cal_ok)} datasets={list(h5ad_map)}", flush=True)

    inf_rows: list[dict] = []
    for _, row in cal_ok.iterrows():
        mir = str(row["mirna"])
        budget = float(row["importance_budget"])
        wpath = Path(str(row["weights_path"]))
        if not wpath.is_file():
            wpath = C.WEIGHTS_DIR / f"{mir.replace('/', '_')}.csv"
        wdf = pd.read_csv(wpath)
        weights = dict(zip(wdf["gene"].astype(str), wdf["weight"].astype(float)))
        panel = cm.sc_genes_for_target(mir, target_info[mir]["genes"], sc_features)
        for ds_name, h5path in h5ad_map.items():
            print(f"  {mir} @ {ds_name} ...", flush=True)
            inf_rows.append(
                assess_dataset(
                    Path(h5path),
                    ds_name,
                    mir,
                    panel,
                    weights,
                    budget,
                    mapping_path=C.GENE_MAPPING_PATH,
                )
            )

    out_path = C.TABLES_DIR / "inference_datasets.csv"
    pd.DataFrame(inf_rows).to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(inf_rows)}", flush=True)

    if not args.no_rebuild_config:
        eligible = list(config["eligible_mirs"])
        cfg = build_robustness_config(
            eligible,
            target_info,
            cal,
            inference_datasets=pd.DataFrame(inf_rows),
        )
        C.ROBUSTNESS_CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {C.ROBUSTNESS_CONFIG_PATH}", flush=True)


if __name__ == "__main__":
    main()
