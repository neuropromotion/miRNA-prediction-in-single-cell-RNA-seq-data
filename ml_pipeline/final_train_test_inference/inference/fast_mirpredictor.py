"""Fast drop-in replacement for ``miRPredictor.predict(adata)``.

Same inputs, same outputs, same models; only the execution path changes:

* the count matrix stays sparse (no dense float64 copies of all genes x all cells);
* pseudobulk sums are one sparse matrix product instead of a Python loop per cell;
* the K1 KNN reference (neighbour index + imputed reference) is built once per object;
* K1 imputation is vectorised and only fills the genes the models actually use;
* robustness gating scans the matrix once instead of once per miRNA;
* model weights are loaded once and kept in RAM, so a second dataset in the same
  session does not re-read them from disk; the next miRNA is prefetched while the
  current one predicts;
* TabPack runs in batches, and base models with zero stack weight are skipped.

Usage (put this file next to ``mirpredictor.py``)::

    from fast_mirpredictor import FastMiRPredictor
    mp = FastMiRPredictor(device="cuda")
    adata = mp.predict(adata, groupby="CellType", embedding_key="X_pca_harmony")
    print(mp.last_timings)          # seconds per stage

``predict_csv`` and every other method are inherited unchanged.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import stack_predictor as _sp  # noqa: E402
from constants import STACK_MODELS, resolve_gene_mapping_path  # noqa: E402
from mirpredictor import ALLOWED_PSEUDOBULK_K, miRPredictor, tqdm  # noqa: E402

__all__ = ["FastMiRPredictor"]


# ----------------------------------------------------------------------------
# timing helper
# ----------------------------------------------------------------------------
class _Timer:
    def __init__(self) -> None:
        self.t: "OrderedDict[str, float]" = OrderedDict()
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def __call__(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, time.perf_counter() - t0)

    def add(self, key: str, dt: float) -> None:
        with self._lock:
            self.t[key] = self.t.get(key, 0.0) + dt

    def as_series(self) -> pd.Series:
        return pd.Series(self.t, name="seconds").round(3)


# ----------------------------------------------------------------------------
# model runners (loaded once, kept in RAM)
# ----------------------------------------------------------------------------
def _resolve_torch_device(device: str):
    import torch

    dev = str(device)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"
    return torch.device(dev)


class _TabPackRunner:
    """Same maths as ``shared.tabpack_trainer.predict_tabpack``, built once."""

    def __init__(self, model_dir: Path) -> None:
        import torch
        from shared.tabpack_trainer import (
            INFERENCE_BUNDLE,
            _build_ensemble_model,
            _ensure_tabpack_on_path,
            _suppress_sklearn_pickle_version_warnings,
            _tabpack_root,
        )

        _ensure_tabpack_on_path(_tabpack_root())
        bundle_path = Path(model_dir) / INFERENCE_BUNDLE
        if not bundle_path.exists():
            meta_path = Path(model_dir) / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                alt = Path(meta.get("exp_dir", "")) / INFERENCE_BUNDLE
                if alt.exists():
                    bundle_path = alt
        if not bundle_path.exists():
            raise FileNotFoundError(f"TabPack inference bundle missing under {model_dir}")
        with _suppress_sklearn_pickle_version_warnings():
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        model, weights, _ = _build_ensemble_model(bundle, "cpu")
        self.model = model
        self.weights = np.asarray(weights, dtype=np.float64)
        self.label_stats = bundle.get("regression_label_stats")
        # only what _transform_raw_x needs (drop the 32 member state dicts)
        self.tbundle = {
            k: bundle.get(k)
            for k in ("bin_extractor", "num_preprocessor", "n_num_features", "cat_cardinalities")
        }
        del bundle

    def predict(self, x: np.ndarray, device, batch_size: int, offload: bool) -> np.ndarray:
        import torch
        from shared.tabpack_trainer import _transform_raw_x

        x_num, x_cat = _transform_raw_x(self.tbundle, x)
        n = (x_num if x_num is not None else x_cat).shape[0]
        self.model.to(device)
        outs = []
        try:
            with torch.inference_mode():
                w = torch.as_tensor(self.weights, device=device, dtype=torch.get_default_dtype())
                w = w / w.sum()
                for s in range(0, n, batch_size):
                    xn = None if x_num is None else torch.as_tensor(
                        x_num[s:s + batch_size], device=device, dtype=torch.get_default_dtype())
                    xc = None if x_cat is None else torch.as_tensor(
                        x_cat[s:s + batch_size], device=device, dtype=torch.long)
                    y = self.model(xn, xc).squeeze(-1).float()  # (P, B)
                    y = (y * w.to(y.dtype)[:, None]).sum(0)
                    if self.label_stats is not None:
                        y = y * float(self.label_stats["std"]) + float(self.label_stats["mean"])
                    outs.append(y.detach().cpu().numpy().astype(np.float64))
        finally:
            if offload and device.type != "cpu":
                self.model.to("cpu")
        out = np.concatenate(outs) if outs else np.zeros(0)
        return np.maximum(out, 0.0)


def _tabm_bundle_cls():
    try:
        from final_train_test_inference.train.dl_trainers import _tabm_bundle_cls as f
    except ImportError:  # pragma: no cover
        from dl_trainers import _tabm_bundle_cls as f
    return f()


class _TabMRunner:
    def __init__(self, model_dir: Path) -> None:
        self.bundle = _tabm_bundle_cls().load(Path(model_dir), device="cpu")

    def predict(self, x: np.ndarray, device, batch_size: int, offload: bool) -> np.ndarray:
        b = self.bundle
        b.model.to(device)
        b.device = device
        try:
            pred = b.predict(x, batch_size=batch_size)
        finally:
            if offload and device.type != "cpu":
                b.model.to("cpu")
                b.device = _resolve_torch_device("cpu")
        return np.clip(np.asarray(pred, dtype=np.float64), 0.0, None)


class _XGBRunner:
    def __init__(self, model_dir: Path) -> None:
        self.model = _sp._model_trainers.load_artifact("xgb_optuna", model_dir.name)

    def predict(self, x: np.ndarray, device, batch_size: int, offload: bool) -> np.ndarray:
        return _sp._model_trainers.predict_one("xgb_optuna", self.model, x)


_RUNNERS = {"tabpack": _TabPackRunner, "tabm": _TabMRunner, "xgb_optuna": _XGBRunner}


class _ModelBank:
    """Per-miRNA stack (fit + three base models) with RAM cache and prefetch."""

    def __init__(self, weights_dir: Path, device: str, *, cache: bool, keep_on_device: bool,
                 batch_size: int, prefetch: bool, timer_ref) -> None:
        self.weights_dir = Path(weights_dir)
        self._device_str = str(device)
        self._device = None
        self.cache = cache
        self.offload = not keep_on_device
        self.batch_size = int(batch_size)
        self._timer_ref = timer_ref  # callable -> current _Timer
        self._store: dict[str, dict] = {}
        self._pending: dict[str, Future] = {}
        self._pool = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._lock = threading.Lock()

    @property
    def device(self):
        if self._device is None:  # import torch only when a model actually runs
            self._device = _resolve_torch_device(self._device_str)
        return self._device

    def _needed_models(self, fit) -> tuple[str, ...]:
        if fit.fallback_best_solo:
            return (fit.best_solo_model,)
        return tuple(m for m in STACK_MODELS if fit.ridge_coef.get(m, 0.0) != 0.0)

    def _load(self, mirna: str) -> dict:
        t0 = time.perf_counter()
        fit = _sp.fit_from_dict(json.loads((self.weights_dir / f"{mirna}.json").read_text(encoding="utf-8")))
        runners = {}
        for m in self._needed_models(fit):
            runners[m] = _RUNNERS[m](Path(_sp._model_trainers.model_dir(m, mirna)))
        entry = {"fit": fit, "runners": runners}
        self._timer_ref().add("model_load", time.perf_counter() - t0)
        return entry

    def prefetch(self, mirna: str) -> None:
        if self._pool is None:
            return
        with self._lock:
            if mirna in self._store or mirna in self._pending:
                return
            self._pending[mirna] = self._pool.submit(self._load, mirna)

    def get(self, mirna: str) -> dict:
        with self._lock:
            entry = self._store.get(mirna)
            fut = self._pending.pop(mirna, None)
        if entry is not None:
            return entry
        if fut is not None:
            t0 = time.perf_counter()
            entry = fut.result()
            self._timer_ref().add("model_load_wait", time.perf_counter() - t0)
        else:
            entry = self._load(mirna)
        if self.cache:
            with self._lock:
                self._store[mirna] = entry
        return entry

    def predict(self, mirna: str, x: np.ndarray) -> np.ndarray:
        entry = self.get(mirna)
        fit = entry["fit"]
        cols = []
        for m in STACK_MODELS:
            if m in entry["runners"]:
                t0 = time.perf_counter()
                cols.append(entry["runners"][m].predict(x, self.device, self.batch_size, self.offload))
                self._timer_ref().add(f"predict_{m}", time.perf_counter() - t0)
            else:  # weight is exactly zero (or fallback to another solo model)
                cols.append(np.zeros(x.shape[0], dtype=np.float64))
        return _sp.apply_fit(fit, np.column_stack(cols), STACK_MODELS)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# ----------------------------------------------------------------------------
# vectorised zero-imputation (same rule as mirpredictor._knn_impute_zeros_cpu)
# ----------------------------------------------------------------------------
def _impute_zeros(q: np.ndarray, neigh: np.ndarray, donor: np.ndarray) -> np.ndarray:
    """q: (c, m) float32, neigh: (c, k) ints into donor rows, donor: (n_ref, m) float32.

    Zero entries of q are replaced by the mean of the non-zero donor values of the
    k neighbours (0 if all neighbours are zero). Modifies and returns q.
    """
    d = donor[neigh]  # (c, k, m)
    sums = d.sum(axis=1)  # float32, same accumulation order as np.nansum over k
    counts = np.count_nonzero(d, axis=1)
    imputed = np.zeros(sums.shape, dtype=np.float32)
    np.divide(sums, counts, out=imputed, where=counts > 0)
    mask = q == 0
    q[mask] = imputed[mask]
    return q


# ----------------------------------------------------------------------------
# the predictor
# ----------------------------------------------------------------------------
class FastMiRPredictor(miRPredictor):
    """``miRPredictor`` with a faster ``predict(adata)``; results are the same."""

    def __init__(self, *args, cache_models: bool = True, keep_models_on_device: bool = False,
                 batch_size: int = 8192, prefetch: bool = True, chunk_cells: int = 2048,
                 n_threads: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cache_models = bool(cache_models)
        self._keep_on_device = bool(keep_models_on_device)
        self._batch_size = int(batch_size)
        self._prefetch = bool(prefetch)
        self._chunk = int(chunk_cells)
        import os
        self._n_threads = int(n_threads or os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
        self._gene_sets_cache: dict[str, tuple[set, dict]] = {}
        self._presence_cache = None
        self._fast_ref: dict | None = None
        self._bank: _ModelBank | None = None
        self._timer = _Timer()
        self.last_timings: pd.Series | None = None

        std = list(self.standard_mrna)
        self._std_pos = {g: i for i, g in enumerate(std)}
        gl = self.gene_lengths.drop_duplicates("gene_id").set_index("gene_id")["gene_length_kb"]
        share = sorted(set(std) & set(gl.index))  # same order as miRPredictor._tpm
        self._share_pos = np.array([self._std_pos[g] for g in share], dtype=np.int64)
        self._len_kb = gl.loc[share].to_numpy(dtype=np.float64)
        self._std_to_share = np.full(len(std), -1, dtype=np.int64)
        self._std_to_share[self._share_pos] = np.arange(len(share))

    # ------------------------------------------------------------------ helpers
    def _gene_sets(self, mapping_path=None) -> tuple[set, dict]:
        path = str(resolve_gene_mapping_path(mapping_path))
        if path not in self._gene_sets_cache:
            genes = pd.read_csv(path)
            all_names = set(genes["feature_id"].tolist()) | set(genes["feature_name"].tolist())
            m = genes.dropna(subset=["feature_name", "feature_id"]).drop_duplicates(subset=["feature_name"])
            self._gene_sets_cache[path] = (all_names, dict(zip(m["feature_name"], m["feature_id"])))
        return self._gene_sets_cache[path]

    def _detect_gene_axis(self, data):  # cached CSV read; same rule as the parent
        all_gene_names, _ = self._gene_sets(None)
        index_hits = len(set(data.index) & all_gene_names)
        col_hits = len(set(data.columns) & all_gene_names)
        if index_hits == 0 and col_hits == 0:
            raise ValueError("Genes were not found neither in columns nor in index")
        return "index" if index_hits > col_hits else "columns"

    # robustness gate: the parent rescans the whole gene table for every miRNA;
    # here the present / non-zero sets are computed once per measured table.
    def _presence_sets(self, mat) -> tuple[set, set]:
        c = self._presence_cache
        if c is not None and c[0] is mat:
            return c[1], c[2]
        present = set(map(self._normalize_ensg, mat.index.astype(str)))
        nz = (mat != 0).any(axis=1)
        nonzero = {self._normalize_ensg(g) for g, flag in nz.items() if bool(flag)}
        self._presence_cache = (mat, present, nonzero)
        return present, nonzero

    def _absent_feature_stats(self, measured_ensg_genes_x_cells, features) -> dict:
        panel = [self._normalize_ensg(g) for g in features]
        n_features = len(panel)
        if n_features == 0:
            return {"n_features": 0, "n_missing": 0, "n_all_zero": 0,
                    "n_present_nonzero": 0, "n_absent": 0, "pct_absent": float("nan")}
        present, nonzero = self._presence_sets(measured_ensg_genes_x_cells)
        n_missing = sum(1 for g in panel if g not in present)
        n_all_zero = sum(1 for g in panel if g in present and g not in nonzero)
        n_absent = n_missing + n_all_zero
        return {"n_features": n_features, "n_missing": n_missing, "n_all_zero": n_all_zero,
                "n_present_nonzero": n_features - n_absent, "n_absent": n_absent,
                "pct_absent": 100.0 * n_absent / n_features}

    def _absent_importance_for_panel(self, measured_ensg_genes_x_cells, panel, mir):
        present, nonzero = self._presence_sets(measured_ensg_genes_x_cells)
        panel_n = [self._normalize_ensg(g) for g in panel]
        absent_set = {g for g in panel_n if g not in present or g not in nonzero}
        wmap = self._gene_weights_for_mir(mir) or {}
        w_absent = sum(wmap.get(g, 0.0) for g in absent_set)
        panel_set = set(panel_n)
        unknown = [g for g in absent_set if g not in wmap and g in panel_set]
        if unknown and wmap:
            w_absent += len(unknown) * (sum(wmap.values()) / len(wmap))
        return float(w_absent), absent_set

    def _bank_get(self) -> _ModelBank:
        if self._bank is None:
            base = self._get_predictor()  # sets FINAL_DEVICE / FINAL_MODELS_ROOT like the parent
            self._bank = _ModelBank(
                base._weights_dir, self._device, cache=self._cache_models,
                keep_on_device=self._keep_on_device, batch_size=self._batch_size,
                prefetch=self._prefetch, timer_ref=lambda: self._timer,
            )
        return self._bank

    def clear_model_cache(self) -> None:
        if self._bank is not None:
            self._bank.clear()

    # ------------------------------------------------------------- public API
    def predict(self, adata, mirnas=None, mapping_path=None, knn_ref_path=None, knn_k=5,
                show_missing_report=False, inplace: bool = False, *, groupby=None,
                embedding_key=None, embedding_dims=None, n_hvg: int = 2000, n_pca: int = 30):
        self._timer = T = _Timer()
        t_all = time.perf_counter()
        with T("extract"):
            ext = self._fast_extract(adata, mapping_path)
        if ext is None:  # unusual gene orientation: use the reference implementation
            warnings.warn("FastMiRPredictor: falling back to miRPredictor.predict", UserWarning)
            return super().predict(adata, mirnas=mirnas, mapping_path=mapping_path,
                                   knn_ref_path=knn_ref_path, knn_k=knn_k,
                                   show_missing_report=show_missing_report, inplace=inplace,
                                   groupby=groupby, embedding_key=embedding_key,
                                   embedding_dims=embedding_dims, n_hvg=n_hvg, n_pca=n_pca)
        with T("extract"):
            celltypes = self._resolve_adata_clusters(adata, groupby=groupby)
            pca_coords = self._resolve_adata_pca(adata, embedding_key=embedding_key,
                                                 embedding_dims=embedding_dims)
        pred = self._fast_full_inference(ext, celltypes, pca_coords, mirnas=mirnas,
                                         knn_ref_path=knn_ref_path, knn_k=knn_k, n_hvg=n_hvg,
                                         n_pca=n_pca, show_missing_report=show_missing_report)
        with T("attach_to_adata"):
            out = self._attach_predictions_to_adata(adata, pred, inplace=inplace)
            out.uns["mirna_prediction"]["groupby"] = str(celltypes.name)
            out.uns["mirna_prediction"]["embedding_key"] = (
                str(pca_coords.attrs.get("obsm_key")) if pca_coords is not None else "computed_pca"
            )
        T.add("TOTAL", time.perf_counter() - t_all)
        self.last_timings = T.as_series()
        if self.log:
            print("Timings (s):", ", ".join(f"{k}={v:.1f}" for k, v in self.last_timings.items()))
        return out

    # ------------------------------------------------------------ extraction
    def _fast_extract(self, adata, mapping_path):
        counts_matrix, gene_labels, source = self._resolve_adata_counts(adata)
        barcodes = np.asarray(adata.obs_names).astype(str)
        var = adata.raw.var if source == "raw" else adata.var
        if "gene_ids" in var.columns:
            gene_labels = var["gene_ids"].astype(str).tolist()
        labels = np.asarray(gene_labels).astype(str)

        X = sp.csr_matrix(counts_matrix, dtype=np.float64) if sp.issparse(counts_matrix) \
            else sp.csr_matrix(np.asarray(counts_matrix, dtype=np.float64))
        X.eliminate_zeros()
        if pd.Index(labels).duplicated().any():  # parent: groupby(level=0).sum() -> sorted labels
            uniq, inv = np.unique(labels, return_inverse=True)
            agg = sp.csr_matrix((np.ones(len(labels)), (np.arange(len(labels)), inv)),
                                shape=(len(labels), len(uniq)))
            X = (X @ agg).tocsr()
            labels = uniq

        all_names, _ = self._gene_sets(None)
        bc_hits = len(set(barcodes) & all_names)
        lab_hits = len(set(labels) & all_names)
        if bc_hits == 0 and lab_hits == 0:
            raise ValueError("Genes were not found neither in columns nor in index")
        if bc_hits > 0:  # barcodes look like gene names: let the reference code decide
            return None

        # Gene mapping is delegated to the parent's own methods, run on a one-row proxy
        # (values = column numbers / non-zero flags). Any change to _prepare_input,
        # _replace_genes_names or _needs_symbol_to_ens_mapping is picked up automatically.
        std_src, measured = self._source_maps(labels, barcodes[0], X, mapping_path)
        have = np.where(std_src >= 0)[0]
        sel = sp.csr_matrix((np.ones(len(have)), (std_src[have], have)),
                            shape=(X.shape[1], len(self.standard_mrna)))
        std_counts = (X @ sel).tocsr()
        return {"barcodes": barcodes, "std_counts": std_counts, "measured": measured,
                "n_genes_in": int(X.shape[1])}

    def _source_maps(self, labels, barcode0, X, mapping_path):
        """Column bookkeeping via the parent's gene-mapping code on a 1-row proxy.

        Returns
          std_src  : for each standard gene, the source column in X (-1 = absent),
                     exactly as _prepare_input would pick it;
          measured : genes x 1 table of non-zero flags with the index that
                     _expression_to_ensg_counts would produce (robustness gate input).
        """
        labels = [str(l) for l in labels]
        n = len(labels)
        proxy = pd.DataFrame(np.arange(1, n + 1, dtype=np.float64)[None, :],
                             index=[barcode0], columns=labels)
        std = self._prepare_input(proxy, mapping_path=mapping_path)  # standard genes x 1
        std_src = np.rint(std.iloc[:, 0].to_numpy()).astype(np.int64) - 1
        col_nz = (np.bincount(X.indices, minlength=n) > 0).astype(np.float64)
        flags = pd.DataFrame(col_nz[None, :], index=[barcode0], columns=labels)
        measured = self._expression_to_ensg_counts(flags, mapping_path=mapping_path)
        measured = measured.iloc[:, :1].copy()
        measured.columns = ["nz"]
        return std_src, measured

    # ------------------------------------------------------------------ TPM
    def _log_tpm_rows(self, counts_rows) -> np.ndarray:
        """(c x n_std) counts (sparse or dense) -> (c x n_share) float64 log2(TPM+1)."""
        dense = counts_rows[:, self._share_pos]
        dense = dense.toarray() if sp.issparse(dense) else np.asarray(dense, dtype=np.float64)
        rpk = dense / self._len_kb[None, :]
        lib = rpk.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tpm = rpk / lib[:, None] * 1e6
        tpm[np.isnan(tpm)] = 0.0
        return np.log2(tpm + 1)

    def _run_row_chunks(self, n: int, fn) -> None:
        """fn(start, stop) over row blocks; numpy releases the GIL, so threads scale."""
        step = max(1, min(self._chunk, -(-n // max(1, self._n_threads))))
        spans = [(s, min(s + step, n)) for s in range(0, n, step)]
        if self._n_threads <= 1 or len(spans) == 1:
            for s, e in spans:
                fn(s, e)
            return
        with ThreadPoolExecutor(max_workers=self._n_threads) as pool:
            for f in [pool.submit(fn, s, e) for s, e in spans]:
                f.result()

    # ------------------------------------------------------------- K1 branch
    def _fast_knn_reference(self, knn_ref_path, knn_k):
        ref_df = self.load_knn_reference(path=knn_ref_path)
        k = min(knn_k, max(1, ref_df.shape[0] - 1))
        key = (str(self._knn_ref_path), k)
        if self._fast_ref is not None and self._fast_ref["key"] == key:
            return self._fast_ref
        from sklearn.neighbors import NearestNeighbors

        xr = ref_df[list(self.standard_mrna)].to_numpy(dtype=np.float32)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1).fit(xr)
        _, ind = nn.kneighbors(xr)
        filled = xr.copy()
        for s in range(0, len(xr), 256):
            _impute_zeros(filled[s:s + 256], ind[s:s + 256], xr)
        self._fast_ref = {"key": key, "nn": nn, "filled": filled}
        return self._fast_ref

    def _k1_matrix(self, std_counts, need_pos, knn_ref_path, knn_k) -> np.ndarray:
        T = self._timer
        with T("k1_reference"):
            ref = self._fast_knn_reference(knn_ref_path, knn_k)
        n_std = len(self.standard_mrna)
        in_share = self._std_to_share[need_pos] >= 0
        need_share_std = need_pos[in_share]
        donor = ref["filled"][:, need_share_std]
        n = std_counts.shape[0]
        out = np.zeros((n, len(need_pos)), dtype=np.float32)
        q_all = np.zeros((n, n_std), dtype=np.float32) if n * n_std <= 500_000_000 else None  # <= 2 GB, else per chunk

        def _tpm_block(s, e):
            lt = self._log_tpm_rows(std_counts[s:e])
            q_all[s:e, self._share_pos] = lt

        if q_all is not None:
            with T("k1_tpm"):
                self._run_row_chunks(n, _tpm_block)
        for s in range(0, n, self._chunk):
            e = min(s + self._chunk, n)
            if q_all is not None:
                q = q_all[s:e]
            else:
                with T("k1_tpm"):
                    q = np.zeros((e - s, n_std), dtype=np.float32)
                    q[:, self._share_pos] = self._log_tpm_rows(std_counts[s:e])
            with T("k1_knn_search"):
                _, ind = ref["nn"].kneighbors(q)
            with T("k1_impute"):
                qn = np.ascontiguousarray(q[:, need_share_std])
                for t in range(0, len(qn), 256):
                    _impute_zeros(qn[t:t + 256], ind[t:t + 256], donor)
                out[s:e, in_share] = qn
        del q_all
        return out

    # -------------------------------------------------------- pseudobulk branch
    def _emb_from_counts(self, counts_rows, n_hvg, n_pca, gene_chunk: int = 1024):
        """Parent's HVG + PCA embedding, without densifying all genes at once.

        counts_rows: (n_cells x n_std) sparse counts of one cluster. Variances are
        computed per gene block with the same numpy calls as the parent
        (log1p(count / lib * 1e4) then np.var along cells), so the HVG choice and the
        PCA input are identical.
        """
        from sklearn.decomposition import PCA

        n_cells, n_genes = counts_rows.shape
        lib = np.asarray(counts_rows.sum(axis=1)).ravel()  # integer counts: exact
        lib = np.where(lib == 0, 1.0, lib)
        csc = counts_rows.tocsc()

        def block(cols):  # genes x cells, like the parent's counts_gc[cols]
            return np.log1p(csc[:, cols].toarray().T / lib * 1e4)

        variances = np.empty(n_genes, dtype=np.float64)

        def _var_block(g0, g1):
            cols = np.arange(g0, g1)
            variances[g0:g1] = np.var(block(cols), axis=1)

        spans = [(g, min(g + gene_chunk, n_genes)) for g in range(0, n_genes, gene_chunk)]
        if self._n_threads > 1:
            with ThreadPoolExecutor(max_workers=self._n_threads) as pool:
                for f in [pool.submit(_var_block, a, b) for a, b in spans]:
                    f.result()
        else:
            for a, b in spans:
                _var_block(a, b)
        n_hvg_eff = min(n_hvg, n_genes)
        top_idx = np.argpartition(variances, -n_hvg_eff)[-n_hvg_eff:]
        x_hvg = block(top_idx)
        n_components = max(1, min(n_pca, x_hvg.shape[0], n_cells))
        return PCA(n_components=n_components, random_state=0).fit_transform(x_hvg.T)

    def _pb_matrix(self, std_counts, barcodes, celltypes, K, need_pos, pca_coords,
                   n_hvg, n_pca, emb_cache) -> tuple[np.ndarray, np.ndarray]:
        T = self._timer
        n_cells = len(barcodes)
        out = np.zeros((n_cells, len(need_pos)), dtype=np.float32)
        valid = np.zeros(n_cells, dtype=bool)
        in_share = self._std_to_share[need_pos] >= 0
        share_cols = self._std_to_share[need_pos[in_share]]
        row_of = {b: i for i, b in enumerate(barcodes)}
        ct_series = celltypes.dropna()
        for cell_type in ct_series.unique():
            type_barcodes = [bc for bc in ct_series.index[ct_series == cell_type].astype(str)
                             if bc in row_of]
            if not type_barcodes or len(type_barcodes) < K:
                continue
            rows = np.fromiter((row_of[b] for b in type_barcodes), dtype=np.int64,
                               count=len(type_barcodes))
            with T(f"pb_neighbors"):
                if pca_coords is not None:
                    emb = pca_coords.reindex(type_barcodes)
                    if emb.isna().any().any():
                        raise ValueError(f"PCA coordinates missing for some cells in '{cell_type}'.")
                    emb = emb.to_numpy(dtype=np.float32)
                else:
                    if cell_type not in emb_cache:
                        emb_cache[cell_type] = self._emb_from_counts(std_counts[rows], n_hvg, n_pca)
                    emb = emb_cache[cell_type]
                idx = self._knn_neighbor_indices_from_embedding(emb, K=K)
            with T("pb_sum"):
                n = len(rows)
                agg = sp.csr_matrix((np.ones(n * K), (np.repeat(np.arange(n), K), idx.ravel())),
                                    shape=(n, n))
                pb = (agg @ std_counts[rows]).tocsr()
            with T("pb_tpm"):
                def _tpm_block(s, e, pb=pb, rows=rows):
                    lt = self._log_tpm_rows(pb[s:e])
                    blk = np.zeros((e - s, len(need_pos)), dtype=np.float32)
                    blk[:, in_share] = lt[:, share_cols]
                    out[rows[s:e]] = blk
                self._run_row_chunks(n, _tpm_block)
            valid[rows] = True
        return out, valid

    # ------------------------------------------------------------ orchestration
    def _fast_full_inference(self, ext, celltypes, pca_coords, *, mirnas, knn_ref_path, knn_k,
                             n_hvg, n_pca, show_missing_report) -> pd.DataFrame:
        T = self._timer
        requested = self._resolve_mirna_subset(mirnas)
        barcodes = ext["barcodes"]
        std_counts = ext["std_counts"]
        print(f"Full inference: {len(barcodes)} cells, {ext['n_genes_in']} genes, "
              f"{len(requested)} eligible miRNAs")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with T("robustness_gate"):
                runnable, skipped, skip_detail = self._filter_by_robustness(ext["measured"], requested)
            print(f"Predicting {len(runnable)}/{len(requested)} miRNAs; "
                  f"{len(skipped)}/{len(requested)} blocked by robustness gate")

            preds = np.full((len(barcodes), len(requested)), np.nan, dtype=np.float64)
            col_of = {m: i for i, m in enumerate(requested)}
            celltypes = celltypes.reindex(pd.Index(barcodes))
            by_cohort: dict[str, list[str]] = {}
            for mir in runnable:
                by_cohort.setdefault(str(self._target_info[mir]["assigned_cohort"]), []).append(mir)
            order = ["K1"] + [f"K{k}" for k in sorted(ALLOWED_PSEUDOBULK_K)]
            plan = [(c, by_cohort[c]) for c in order if by_cohort.get(c)]
            flat = [m for _, ms in plan for m in ms]
            bank = self._bank_get() if flat else None
            if bank is not None:
                bank.prefetch(flat[0])
            if show_missing_report:
                print(f"Note: {len(self.standard_mrna) - len(self._share_pos)} standard genes "
                      "have no length and are filled with 0.")

            emb_cache: dict = {}
            pbar = tqdm(total=len(flat), desc="Predicting miRNAs", unit="mir")
            pos = 0
            try:
                for cohort, mirs in plan:
                    genes_needed: list[str] = []
                    seen = set()
                    for m in mirs:
                        for g in self._target_info[m]["genes"]:
                            if g not in seen:
                                seen.add(g)
                                genes_needed.append(g)
                    missing = [g for g in genes_needed if g not in self._std_pos]
                    if missing:
                        raise KeyError(f"missing {len(missing)} features, e.g. {missing[:3]}")
                    need_pos = np.array([self._std_pos[g] for g in genes_needed], dtype=np.int64)
                    need_col = {g: i for i, g in enumerate(genes_needed)}

                    if cohort == "K1":
                        mat = self._k1_matrix(std_counts, need_pos, knn_ref_path, knn_k)
                        rows = np.arange(len(barcodes))
                    else:
                        K = int(cohort[1:])
                        mat, valid = self._pb_matrix(std_counts, barcodes, celltypes, K, need_pos,
                                                     pca_coords, n_hvg, n_pca, emb_cache)
                        rows = np.where(valid)[0]
                        mat = mat[rows]
                    with T("select_features"):
                        mat = np.asfortranarray(mat)  # per-miRNA column gathers become contiguous copies
                    for m in mirs:
                        pos += 1
                        if pos < len(flat):
                            bank.prefetch(flat[pos])
                        if len(rows) == 0:
                            pbar.update(1)
                            continue
                        cols = [need_col[g] for g in self._target_info[m]["genes"]]
                        with T("select_features"):
                            x = np.ascontiguousarray(mat[:, cols])
                        preds[rows, col_of[m]] = bank.predict(m, x)
                        pbar.update(1)
                    del mat
            finally:
                pbar.close()

        out = pd.DataFrame(preds, index=pd.Index(barcodes), columns=requested)
        out.attrs["predicted_mirnas"] = list(runnable)
        out.attrs["skipped_mirnas"] = list(skipped)
        out.attrs["skip_detail"] = skip_detail
        return out
