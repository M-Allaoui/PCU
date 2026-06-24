#!/usr/bin/env python3
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from PCU import PCU, PCUConfig

INT_KEYS = {
    "dz", "enc_width", "enc_depth", "K", "epochs", "batch_size",
    "noise_eval_copies", "m_parts", "seed",
    "tiny_eval_copies", "score_stats_batch"
}
FLOAT_KEYS = {"dropout","sigma_min","sigma_max","lr","weight_decay","rank_margin",
              "w_rank","w_scale","w_vicreg","alpha","beta","gamma","tiny_sigma","ema_decay",
              "sigma_max_paper"}

def log(msg: str):
    print(msg, flush=True)

def to_native(obj):
    import numpy as _np
    if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_native(v) for v in obj]
    if isinstance(obj, _np.generic): return obj.item()
    if hasattr(obj, "tolist"): return obj.tolist()
    return obj

def coerce_types(cfg: dict, device_override: str | None):
    cfg = cfg.copy()
    for k in list(cfg.keys()):
        if k in INT_KEYS and cfg[k] is not None:   cfg[k] = int(cfg[k])
        if k in FLOAT_KEYS and cfg[k] is not None: cfg[k] = float(cfg[k])
    if device_override:
        cfg["device"] = device_override
    cfg.setdefault("device", "cuda")
    cfg.setdefault("ema_decay", 0.99)
    return cfg

def _subsample_normals(rng: np.random.Generator, Xn: np.ndarray, train_norm_frac: float, max_train_normals: int | None):
    if train_norm_frac < 1.0:
        n = max(1, int(len(Xn) * train_norm_frac))
        Xn = Xn[rng.permutation(len(Xn))[:n]]
    if max_train_normals is not None and len(Xn) > max_train_normals:
        Xn = Xn[rng.permutation(len(Xn))[:max_train_normals]]
    return Xn

def _load_indices_from_npz(npz_obj):
    keys = set(npz_obj.files)

    def pick(cands):
        for k in cands:
            if k in keys:
                return k, np.asarray(npz_obj[k]).astype(np.int64).ravel()
        return None, None

    k_tr, tr = pick(["train_idx","idx_train","train_indices","train","tr_idx"])
    k_te, te = pick(["test_idx","idx_test","test_indices","test","te_idx"])
    if tr is None or te is None:
        return None, None, None
    return tr, te, {"train_key": k_tr, "test_key": k_te}

def _one_class_from_indices(X, y, train_idx, test_idx, rng, train_norm_frac, max_train_normals):
    Xtr, ytr = X[train_idx], y[train_idx]
    X_train_norm = Xtr[ytr == 0]
    X_train_norm = _subsample_normals(rng, X_train_norm, train_norm_frac, max_train_normals)
    Xte, yte = X[test_idx], y[test_idx]
    return X_train_norm.astype(np.float32), Xte.astype(np.float32), yte.astype(int)

def _fallback_one_class_split_from_Xy(X, y, seed, train_norm_frac=1.0, max_train_normals=None, train_split=0.8):
    rng = np.random.default_rng(seed)
    Xn = X[y == 0]
    Xa = X[y != 0]
    if len(Xn) < 2:
        raise ValueError("Not enough normal samples to create a split.")
    if not (0.0 < train_split < 1.0):
        raise ValueError("--train_split must be in (0,1).")

    perm = rng.permutation(len(Xn))
    n_train = max(1, int(len(Xn) * train_split))
    idx_tr = perm[:n_train]
    idx_te = perm[n_train:]

    X_train_norm = _subsample_normals(rng, Xn[idx_tr], train_norm_frac, max_train_normals)
    X_test_norm  = Xn[idx_te]

    X_test = np.concatenate([X_test_norm, Xa], axis=0) if len(Xa) else X_test_norm.copy()
    y_test = np.concatenate([np.zeros(len(X_test_norm), dtype=int), np.ones(len(Xa), dtype=int)], axis=0) if len(Xa) else np.zeros(len(X_test_norm), dtype=int)
    return X_train_norm.astype(np.float32), X_test.astype(np.float32), y_test.astype(int)

def load_adbench_npz(path, seed, train_norm_frac=1.0, max_train_normals=None, train_split=0.8, split_dir=None):
    """
    Split priority:
      (A) dataset file contains X_train/y_train/X_test/y_test
      (B) dataset file contains train/test indices
      (C) external split_dir contains <stem>_seed<seed>.npz indices
      (D) fallback split (only if A–C missing)
    """
    rng = np.random.default_rng(seed)
    npz_path = Path(path)
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)

    # (A)
    if {"X_train","y_train","X_test","y_test"} <= keys:
        Xtr, ytr = d["X_train"].astype(np.float32), d["y_train"].astype(int)
        Xte, yte = d["X_test"].astype(np.float32),  d["y_test"].astype(int)
        X_train_norm = _subsample_normals(rng, Xtr[ytr == 0], train_norm_frac, max_train_normals)
        return X_train_norm, Xte, yte, {"split_source": "npz_arrays"}

    # must have X,y for indices modes
    if not ({"X","y"} <= keys):
        raise KeyError(f"Unsupported .npz format. Found keys: {sorted(keys)}")
    X, y = d["X"].astype(np.float32), d["y"].astype(int)

    # (B)
    tr, te, meta = _load_indices_from_npz(d)
    if tr is not None and te is not None:
        Xn, Xte, yte = _one_class_from_indices(X, y, tr, te, rng, train_norm_frac, max_train_normals)
        return Xn, Xte, yte, {"split_source": "npz_indices", **meta}

    # (C)
    if split_dir is not None:
        split_file = Path(split_dir) / f"{npz_path.stem}_seed{seed}.npz"
        if split_file.exists():
            s = np.load(split_file, allow_pickle=True)
            tr2, te2, meta2 = _load_indices_from_npz(s)
            if tr2 is None or te2 is None:
                raise KeyError(f"{split_file} missing train/test indices keys. Found: {sorted(s.files)}")
            Xn, Xte, yte = _one_class_from_indices(X, y, tr2, te2, rng, train_norm_frac, max_train_normals)
            return Xn, Xte, yte, {"split_source": "split_dir", "split_file": str(split_file), **meta2}

    # (D)
    Xn, Xte, yte = _fallback_one_class_split_from_Xy(X, y, seed, train_norm_frac, max_train_normals, train_split)
    return Xn, Xte, yte, {"split_source": "fallback_normal_split", "train_split_normals": float(train_split)}

def find_npz_files(data_dir: str, glob_pattern: str):
    base = Path(data_dir)
    return sorted(p for p in base.rglob(glob_pattern) if p.is_file())

def normalize_dataset_arg(ds: str):
    ds = ds.strip()
    if ds.lower().endswith(".npz"):
        p = Path(ds)
        return p.stem, p.name
    return ds, f"{ds}.npz"

def dataset_suffix(stem: str) -> str:
    """
    Converts:
        43_WDBC -> WDBC
        1_ALOI  -> ALOI
    """
    if "_" in stem and stem.split("_", 1)[0].isdigit():
        return stem.split("_", 1)[1]
    return stem


def find_dataset_by_name(data_dir: Path, dataset: str):
    """
    Allows:
      --dataset 43_WDBC
      --dataset WDBC
      --dataset 43_WDBC.npz
      --dataset WDBC.npz
    """
    req_stem = Path(dataset).stem
    all_files = sorted(data_dir.rglob("*.npz"))

    matches = []
    for p in all_files:
        stem = p.stem
        suffix = dataset_suffix(stem)

        if stem.lower() == req_stem.lower():
            matches.append(p)
        elif suffix.lower() == req_stem.lower():
            matches.append(p)
        elif p.name.lower() == dataset.lower():
            matches.append(p)

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise ValueError(
            f"Dataset {dataset} not found under {data_dir}. "
            f"Use exact name such as 43_WDBC or short name such as WDBC."
        )

    raise ValueError(
        f"Dataset name {dataset} matched multiple files: {[str(m) for m in matches]}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dataset", default=None, help="Optional: WDBC or WDBC.npz")
    ap.add_argument("--glob", default="*.npz")
    ap.add_argument("--config_json_default", required=True)
    ap.add_argument("--out_dir", default="pcu_results")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train_norm_frac", type=float, default=1.0)
    ap.add_argument("--max_train_normals", type=int, default=None)
    ap.add_argument("--train_split", type=float, default=0.99, help="Fallback only (used if no official split found).")
    ap.add_argument("--split_dir", type=str, default=None,
                    help="Directory with <dataset>_seed<seed>.npz split index files.")
    args = ap.parse_args()

    log("PCU Clean Evaluation (UAI)")
    log(f"Data:  {args.data_dir}")
    log(f"Seeds: {args.seeds}  (0..{args.seeds-1})")
    if args.split_dir:
        log(f"Split: {args.split_dir}")
    log("------------------------------------------------------------")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"--data_dir not found: {data_dir}")

    cfg_path = Path(args.config_json_default)
    if not cfg_path.exists():
        raise FileNotFoundError(f"--config_json_default not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        base_cfg = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)

    if args.dataset:
        datasets = [find_dataset_by_name(data_dir, args.dataset)]
    else:
        datasets = find_npz_files(str(data_dir), args.glob)
        if not datasets:
            raise ValueError(f"No datasets found under {data_dir} with pattern {args.glob}")

    for npz_path in datasets:
        ds_name = npz_path.stem
        aucs, times, split_sources = [], [], []

        for seed in range(args.seeds):
            Xn, Xte, yte, split_meta = load_adbench_npz(
                str(npz_path), seed=seed,
                train_norm_frac=args.train_norm_frac,
                max_train_normals=args.max_train_normals,
                train_split=args.train_split,
                split_dir=args.split_dir,
            )
            split_sources.append(split_meta.get("split_source", "unknown"))

            cfg_dict = coerce_types(base_cfg, args.device)
            cfg_dict["seed"] = int(seed)

            t0 = time.perf_counter()
            det = PCU(PCUConfig(**cfg_dict)).fit(Xn, verbose=False)
            scores = det.score_samples(Xte)
            runtime_s = float(time.perf_counter() - t0)

            auc = float(roc_auc_score(yte, scores))
            aucs.append(auc)
            times.append(runtime_s)

            log(f"{ds_name} | seed {seed} -> AUROC {auc:.4f} | time={runtime_s:.1f}s")

        auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs, ddof=0))
        t_mean, t_std     = float(np.mean(times)), float(np.std(times, ddof=0))

        log(f"[summary] {ds_name}: AUROC {auc_mean:.4f} ± {auc_std:.4f} | Time {t_mean:.2f}s ± {t_std:.2f}s over {args.seeds} seed(s)")
        log("------------------------------------------------------------")

        with open(out_dir / f"{ds_name}.json", "w") as jf:
            json.dump({
                "dataset": ds_name,
                "npz_path": str(npz_path),
                "AUC_mean": auc_mean,
                "AUC_std": auc_std,
                "runtime_s_mean": t_mean,
                "runtime_s_std": t_std,
                "seeds": int(args.seeds),
                "split_sources_seen": sorted(set(split_sources)),
                "config_file": str(cfg_path),
                "config_used": to_native(cfg_dict),
            }, jf, indent=2)

    log("All done.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)