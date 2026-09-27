"""B4.5 (WORKPLAN.md) - light-to-heavy stage1 model comparison + a couple of extra LightGBM
hyperparameter combos, all measured on the SAME train/val split and SAME full 2-stage pipeline
(stage2 + decode unchanged) so results are directly comparable by validation macro F0.5.

Does NOT modify model.py or train.py - this is a read-only benchmark. Whatever wins, you copy
its config into model.py's P1 dict yourself (per WORKPLAN.md B4.5: "do not swap the production
model without Track D's sign-off").

Usage (same data/workers args as train.py):
    python src/model_bench.py --data ../../sample_dense --workers 5
    python src/model_bench.py --data ../../sample_dense --workers 5 --candidates logreg,rf200,current
"""
import argparse, time, zlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from io_utils import read_tsv, countries_of
from pipeline import build_country
from features import pair_features
from model import stage2_matrix, raw2, fit_stage2, decode_prep, decode_apply, P1 as CURRENT_P1
from evaluate import f05_macro

# Optional heavier candidates - only registered if the library is actually installed, so this
# script never crashes just because catboost/xgboost aren't pip-installed yet. Install with:
#   pip install catboost xgboost
try:
    from catboost import CatBoostClassifier
    HAVE_CATBOOST = True
except ImportError:
    HAVE_CATBOOST = False

try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
    TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAVE_TORCH = False
    TORCH_DEVICE = "cpu"

CFG = dict(max_block=30, max_s1_block=200, topk=30)

# thr/margin sweep - same widened range as train.py's current sweep
THR_GRID    = np.arange(0.30, 0.99, 0.02)
MARGIN_GRID = np.arange(0.00, 0.45, 0.02)


class LGBWrap:
    """Thin wrapper so LightGBM configs share the same .fit/.predict_proba interface as sklearn
    models below, with no eval_set/early stopping (bench = plain fixed-n_estimators fit, matching
    what train.py's OOF loop does)."""
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.m = lgb.LGBMClassifier(**self.params)
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


class Pipeline_LR:
    """StandardScaler + LogisticRegression, same .fit/.predict_proba interface."""
    def fit(self, X, y):
        self.sc = StandardScaler()
        Xs = self.sc.fit_transform(X)
        self.m = LogisticRegression(max_iter=300, class_weight="balanced")
        self.m.fit(Xs, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(self.sc.transform(X))


class MLPWrap:
    """StandardScaler + MLPClassifier (feedforward neural net), same .fit/.predict_proba
    interface as everything else here. Trained on the same engineered tabular features as
    every other candidate - a from-scratch embedding/transformer model on raw text is NOT
    feasible to build and validate safely in the time left, so this is the honest version of
    'try a neural net' that fits the same fair benchmark."""
    def __init__(self, hidden=(64, 32), **params):
        self.hidden = hidden
        self.params = params

    def fit(self, X, y):
        self.sc = StandardScaler()
        Xs = self.sc.fit_transform(X)
        self.m = MLPClassifier(hidden_layer_sizes=self.hidden, activation="relu",
                                alpha=1e-3, learning_rate_init=1e-3, max_iter=300,
                                early_stopping=True, n_iter_no_change=15, **self.params)
        self.m.fit(Xs, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(self.sc.transform(X))


class _TorchNet(nn.Module if HAVE_TORCH else object):
    """Plain feedforward net: Linear -> BatchNorm -> ReLU -> Dropout, stacked, sigmoid output."""
    def __init__(self, n_in, hidden=(128, 64, 32), dropout=0.2):
        super().__init__()
        layers = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchWrap:
    """GPU-trained feedforward neural net (PyTorch), same .fit/.predict_proba interface as
    everything else here. Trained on the SAME engineered tabular features as LightGBM/CatBoost/
    XGBoost - honest apples-to-apples comparison, just a different model family. Uses the GPU
    (RTX 4050 / cuda) when available - orders of magnitude faster than sklearn's CPU-only MLP."""
    def __init__(self, hidden=(128, 64, 32), epochs=40, batch_size=4096, lr=1e-3,
                 dropout=0.2, patience=5):
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.patience = patience

    def fit(self, X, y):
        dev = TORCH_DEVICE
        self.sc = StandardScaler()
        Xs = self.sc.fit_transform(X).astype(np.float32)
        y = y.astype(np.float32)

        # internal train/early-stop split (90/10, random - fine for a benchmark)
        n = len(y)
        rng = np.random.RandomState(0)
        idx = rng.permutation(n)
        n_val = max(1, int(n * 0.1))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        Xt = torch.tensor(Xs[tr_idx], device=dev)
        yt = torch.tensor(y[tr_idx], device=dev)
        Xv = torch.tensor(Xs[val_idx], device=dev)
        yv = torch.tensor(y[val_idx], device=dev)

        self.m = _TorchNet(Xs.shape[1], hidden=self.hidden, dropout=self.dropout).to(dev)
        pos_weight = torch.tensor([(len(yt) - yt.sum()) / max(yt.sum(), 1)], device=dev)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.m.parameters(), lr=self.lr)

        best_val = float("inf"); best_state = None; bad = 0
        n_tr = len(yt)
        for ep in range(self.epochs):
            self.m.train()
            perm = torch.randperm(n_tr, device=dev)
            for s in range(0, n_tr, self.batch_size):
                b = perm[s:s + self.batch_size]
                opt.zero_grad()
                loss = crit(self.m(Xt[b]), yt[b])
                loss.backward()
                opt.step()
            self.m.eval()
            with torch.no_grad():
                vloss = crit(self.m(Xv), yv).item()
            if vloss < best_val - 1e-4:
                best_val = vloss; best_state = {k: v.clone() for k, v in self.m.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.m.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        dev = TORCH_DEVICE
        Xs = self.sc.transform(X).astype(np.float32)
        self.m.eval()
        with torch.no_grad():
            logits = self.m(torch.tensor(Xs, device=dev)).cpu().numpy()
        p1 = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1 - p1, p1], axis=1)


class BlendWrap:
    """Averages the predict_proba of several sub-estimators (built from zero-arg factories).
    Different model families make different mistakes, so blending their probabilities often
    beats any single one - cheap to try since it reuses estimators we've already benchmarked."""
    def __init__(self, factories, weights=None):
        self.factories = factories
        self.weights = weights or [1.0 / len(factories)] * len(factories)

    def fit(self, X, y):
        self.models = []
        for f in self.factories:
            est = f()
            est.fit(X, y)
            self.models.append(est)
        return self

    def predict_proba(self, X):
        p1 = np.zeros(len(X), dtype=np.float64)
        for w, m in zip(self.weights, self.models):
            p1 += w * m.predict_proba(X)[:, 1]
        return np.stack([1 - p1, p1], axis=1)


class CatWrap:
    """CatBoostClassifier, same .fit/.predict_proba interface. verbose=False to keep bench
    output clean; class weighting via auto_class_weights (CatBoost's is_unbalance equivalent)."""
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.m = CatBoostClassifier(auto_class_weights="Balanced", verbose=False, **self.params)
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


class XGBWrap:
    """XGBClassifier, same .fit/.predict_proba interface."""
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.m = XGBClassifier(eval_metric="auc", **self.params)
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


CANDIDATES = {
    # name -> zero-arg factory returning a fresh, unfit estimator with .fit/.predict_proba
    "logreg":       lambda: Pipeline_LR(),
    "rf200":        lambda: RandomForestClassifier(
                        n_estimators=200, max_depth=20, n_jobs=-1,
                        class_weight="balanced", random_state=0),
    "current":      lambda: LGBWrap(**CURRENT_P1),
    "lgb_700_127":  lambda: LGBWrap(**{**CURRENT_P1, "n_estimators": 700, "num_leaves": 127}),
    "lgb_300_31_lr08":  lambda: LGBWrap(**{**CURRENT_P1, "n_estimators": 300, "num_leaves": 31,
                                            "learning_rate": 0.08}),
    "lgb_500_127_lr08": lambda: LGBWrap(**{**CURRENT_P1, "n_estimators": 500, "num_leaves": 127,
                                            "learning_rate": 0.08}),
    "mlp":          lambda: MLPWrap(hidden=(64, 32)),
    "mlp_deep":     lambda: MLPWrap(hidden=(128, 64, 32)),
}

if HAVE_CATBOOST:
    CANDIDATES["catboost"] = lambda: CatWrap(iterations=500, depth=6, learning_rate=0.05)
else:
    print("[model_bench] catboost not installed - skipping 'catboost' candidate "
          "(pip install catboost to enable it)")

if HAVE_XGBOOST:
    CANDIDATES["xgboost"] = lambda: XGBWrap(n_estimators=500, max_depth=6, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8)
else:
    print("[model_bench] xgboost not installed - skipping 'xgboost' candidate "
          "(pip install xgboost to enable it)")

if HAVE_TORCH:
    print(f"[model_bench] torch available, device={TORCH_DEVICE}")
    CANDIDATES["torchnet"] = lambda: TorchWrap(hidden=(128, 64, 32), epochs=40)
    CANDIDATES["torchnet_deep"] = lambda: TorchWrap(hidden=(256, 128, 64, 32), epochs=40)
else:
    print("[model_bench] torch not installed - skipping 'torchnet'/'torchnet_deep' candidates "
          "(pip install torch to enable them)")

# Blends - reuse the factories already registered above, so they only appear once xgboost/
# catboost are actually available.
if HAVE_XGBOOST:
    CANDIDATES["blend_lgb_xgb"] = lambda: BlendWrap(
        [lambda: LGBWrap(**CURRENT_P1), CANDIDATES["xgboost"]])
if HAVE_XGBOOST and HAVE_CATBOOST:
    CANDIDATES["blend_lgb_xgb_cat"] = lambda: BlendWrap(
        [lambda: LGBWrap(**CURRENT_P1), CANDIDATES["xgboost"], CANDIDATES["catboost"]])


def sweep_f05(r1, ro, p2, u1, uo, val_ids, vt):
    prepped = decode_prep(r1, ro, p2)
    best = (-1.0, 0.5, 0.0)
    for thr in THR_GRID:
        for mg in MARGIN_GRID:
            rr, oo = decode_apply(prepped, thr, mg)
            pred = {}
            for i, o in zip(rr, oo):
                s = u1[i]
                if s in val_ids:
                    pred.setdefault(s, set()).add(uo[o])
            sc = f05_macro(pred, vt)
            if sc > best[0]:
                best = (sc, float(thr), float(mg))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--candidates", default=",".join(CANDIDATES.keys()),
                     help="comma-separated subset of: " + ",".join(CANDIDATES.keys()))
    a = ap.parse_args()
    names = a.candidates.split(",")
    for n in names:
        if n not in CANDIDATES:
            ap.error(f"unknown candidate '{n}'. choices: {','.join(CANDIDATES.keys())}")

    # ------------------------------------------------------------------ same data build as train.py
    gt = read_tsv(f"{a.data}/train_ground_truth.tsv")
    truth = {k: {m for m in v.split(",") if m}
             for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    X, Y, S1ID, OID = [], [], [], []
    for c in countries_of(f"{a.data}/train_source1.tsv"):
        s1, oth, cand = build_country(a.data, "train", c, CFG, a.workers)
        if cand is None:
            continue
        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        feats = pair_features(s1, oth, r1, ro, w, a.workers)
        s1ids = np.array(s1.ids, dtype=object)[r1]
        oids  = np.array(oth.ids, dtype=object)[ro]
        y = np.fromiter((o in truth.get(s, ()) for s, o in zip(s1ids, oids)),
                         dtype=np.int8, count=len(r1))
        X.append(feats); Y.append(y); S1ID.append(s1ids); OID.append(oids)

    X = np.vstack(X); Y = np.concatenate(Y)
    S1ID = np.concatenate(S1ID); OID = np.concatenate(OID)
    print(f"total pairs: {len(Y)}")

    u1, r1 = np.unique(S1ID, return_inverse=True)
    uo, ro = np.unique(OID,  return_inverse=True)
    is_val = np.array([zlib.crc32((s + "v").encode()) % 10 < 3 for s in u1])[r1]
    tr     = ~is_val
    fold   = np.array([zlib.crc32((s + "f").encode()) % 2 for s in u1])[r1]
    val_ids = {k for k in truth if zlib.crc32((k + "v").encode()) % 10 < 3}
    vt      = {k: v for k, v in truth.items() if k in val_ids}

    RAW2 = raw2(X)
    results = []

    for name in names:
        print(f"\n=== {name} ===")
        t0 = time.time()

        # out-of-fold p1 on the training portion (same scheme as train.py, needed so stage2
        # doesn't learn from an overfit/in-sample p1 on the training rows)
        p1 = np.zeros(len(Y), dtype=np.float32)
        for f in (0, 1):
            fit_mask  = tr & (fold != f)
            pred_mask = tr & (fold == f)
            est = CANDIDATES[name]()
            est.fit(X[fit_mask], Y[fit_mask])
            p1[pred_mask] = est.predict_proba(X[pred_mask])[:, 1]

        # final stage1 fit on all of tr -> used for val p1 and as the reported train/predict time
        est = CANDIDATES[name]()
        t_fit0 = time.time()
        est.fit(X[tr], Y[tr])
        train_time = time.time() - t_fit0

        t_pred0 = time.time()
        p1[is_val] = est.predict_proba(X[is_val])[:, 1]
        predict_time = time.time() - t_pred0

        # stage2 unchanged - current production config, same as train.py
        X2 = stage2_matrix(r1, ro, p1, RAW2)
        m2 = fit_stage2(X2[tr], Y[tr])
        p2 = m2.predict_proba(X2)[:, 1]

        f05, best_thr, best_mg = sweep_f05(r1, ro, p2, u1, uo, val_ids, vt)
        total_time = time.time() - t0

        print(f"val F0.5={f05:.4f}  thr={best_thr:.2f} margin={best_mg:.2f}  "
              f"stage1_train={train_time:.1f}s  stage1_predict={predict_time:.2f}s  "
              f"total={total_time:.1f}s")
        results.append((name, f05, best_thr, best_mg, train_time, predict_time, total_time))

    print("\n" + "=" * 100)
    print(f"{'model':<20}{'val F0.5':>10}{'thr':>7}{'margin':>8}{'train_s':>10}{'predict_s':>11}{'total_s':>9}")
    for name, f05, thr, mg, tt, pt, tot in sorted(results, key=lambda r: -r[1]):
        print(f"{name:<20}{f05:>10.4f}{thr:>7.2f}{mg:>8.2f}{tt:>10.1f}{pt:>11.2f}{tot:>9.1f}")


if __name__ == "__main__":
    main()
