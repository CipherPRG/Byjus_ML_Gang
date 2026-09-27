"""Train the two-stage matcher on a (dense) training sample and tune the decision threshold for macro F0.5.

Normal mode  — trains on sample_dense, holds out 30 % for validation, sweeps thr/margin, saves models/:
    python src/train.py --data ../../sample_dense --models ../../models --workers 4

Final mode   — retrains on ALL data (no holdout) using a pre-locked thr/margin from an existing config.json,
               saves to models_final/.  Run this ONLY after thr/margin are confirmed from a normal-mode run:
    python src/train.py --data ../../sample_dense --models ../../models_final --final \
        --thr 0.70 --margin 0.20 --workers 4
"""
import argparse, hashlib, json, os, zlib
import numpy as np
import pandas as pd
from io_utils import read_tsv, countries_of
from pipeline import build_country
from features import pair_features, F1
from model import fit_stage1, fit_stage2, stage2_matrix, decode, decode_prep, decode_apply, raw2
from model import best_assignment, fit_ef_decoder, decode_ef_assigned
from evaluate import f05_macro

CFG = dict(max_block=60, max_s1_block=200, topk=60)


def es_half_of(s):
    """ES/REP half of the val set. MD5, NOT crc32: crc32 is affine, so crc32(s+'e') parity is tied to
    crc32(s+'v') (which picks val) and gave a 2:1 split correlated with the val selection."""
    return hashlib.md5((s + "e").encode()).digest()[0] % 2 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    required=True,  help="path to sample_dense/ (or full train/) folder")
    ap.add_argument("--models",  required=True,  help="output directory for model artefacts")
    ap.add_argument("--workers", type=int, default=1)
    # --final: skip val holdout, use all data for training.  Must supply --thr and --margin.
    ap.add_argument("--final",   action="store_true",
                    help="retrain on ALL data (no holdout). Requires --thr and --margin.")
    ap.add_argument("--thr",     type=float, default=None,
                    help="fixed threshold (required with --final, optional otherwise to skip sweep)")
    ap.add_argument("--margin",  type=float, default=None,
                    help="fixed margin  (required with --final, optional otherwise to skip sweep)")
    ap.add_argument("--addr-stop-frac", type=float, default=None,
                    help="skip address words present in more than this fraction of a country's addresses "
                         "when building address keys (e.g. 0.01). Saved into config.json so predict matches.")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="learning-curve check: train on only this fraction of the TRAIN entities (md5 hash); "
                         "val entities are unchanged, so scores stay comparable. Default 1.0 = all.")
    ap.add_argument("--keys-v2", action="store_true",
                    help="extra blocking keys (address number bigrams, number x rare word, compact/website "
                         "names) + ordinal normalisation. Saved into config.json so predict matches.")
    a = ap.parse_args()
    if a.addr_stop_frac is not None:
        CFG["addr_stop_frac"] = a.addr_stop_frac
    if a.keys_v2:
        CFG["keys_v2"] = True
    print(f"CFG = {CFG}")

    if a.final and (a.thr is None or a.margin is None):
        ap.error("--final requires --thr and --margin (copy the best values from a normal-mode run's config.json)")

    os.makedirs(a.models, exist_ok=True)

    # ------------------------------------------------------------------ load ground truth
    gt = read_tsv(f"{a.data}/train_ground_truth.tsv")
    truth = {k: {m for m in v.split(",") if m}
             for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    # ------------------------------------------------------------------ blocking + features
    # Labelled rows = pairs of S1 that have a ground-truth row: trained on / scored, features kept in RAM.
    # Context rows = pairs of rival S1 without a GT row (sample_v3): never trained on or scored; they only
    # provide realistic competition for stage 2 and the decode. Their features go to disk in batches
    # (like predict) and only their stage-1 scores + RAW2 columns are used. No context rows -> old behaviour.
    X, Y, S1ID, OID = [], [], [], []
    C_S1, C_O, C_FILES = [], [], []
    ctx_dir = os.path.join(a.models, "_ctx_tmp")
    n_true = n_found = 0
    for c in countries_of(f"{a.data}/train_source1.tsv"):
        s1, oth, cand = build_country(a.data, "train", c, CFG, a.workers)
        if cand is None:
            continue
        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        del cand
        s1ids = np.array(s1.ids, dtype=object)[r1]
        oids  = np.array(oth.ids, dtype=object)[ro]
        lab = np.fromiter((s in truth for s in s1ids), dtype=bool, count=len(r1))
        feats = pair_features(s1, oth, r1[lab], ro[lab], w[lab], a.workers)
        y = np.fromiter(
            (o in truth[s] for s, o in zip(s1ids[lab], oids[lab])),
            dtype=np.int8, count=int(lab.sum())
        )
        tot = sum(len(truth[i]) for i in s1.ids if i in truth)
        n_ctx = int((~lab).sum())
        print(f"[{c}] S1={len(s1)} others={len(oth)} pairs={int(lab.sum())} "
              f"({lab.sum()/max(len(set(s1ids[lab])),1):.1f}/S1) blocking recall={y.sum()/max(tot,1):.4f}"
              + (f" | rival context pairs={n_ctx:,}" if n_ctx else ""), flush=True)
        n_true += tot; n_found += int(y.sum())
        S1ID.append(s1ids[lab]); OID.append(oids[lab]); X.append(feats); Y.append(y)
        if n_ctx:
            os.makedirs(ctx_dir, exist_ok=True)
            idx = np.flatnonzero(~lab)
            fn = os.path.join(ctx_dir, f"{c}.npy")
            mm = np.lib.format.open_memmap(fn, mode="w+", dtype=np.float32, shape=(len(idx), len(F1)))
            B = 2_000_000
            for s in range(0, len(idx), B):
                j = idx[s:s + B]
                mm[s:s + len(j)] = pair_features(s1, oth, r1[j], ro[j], w[j], a.workers)
                print(f"  [{c}] rival-context features {min(s + B, len(idx)):,}/{len(idx):,}", flush=True)
            mm.flush(); del mm
            C_S1.append(s1ids[~lab]); C_O.append(oids[~lab]); C_FILES.append(fn)
        del s1, oth, r1, ro, w, s1ids, oids

    S1ID = np.concatenate(S1ID); OID = np.concatenate(OID)
    X    = np.vstack(X);          Y   = np.concatenate(Y)
    nL = len(Y)
    CS1 = np.concatenate(C_S1) if C_S1 else np.array([], dtype=object)
    CO  = np.concatenate(C_O) if C_O else np.array([], dtype=object)
    nC = len(CS1)
    print(f"overall blocking recall: {n_found/n_true:.4f}; labelled pairs: {nL:,}; rival context pairs: {nC:,}")

    def ctx_chunks():
        """Yield rival-context feature blocks in row order (disk-backed)."""
        for fn in C_FILES:
            mm = np.load(fn, mmap_mode="r")
            for s in range(0, len(mm), 2_000_000):
                yield np.array(mm[s:s + 2_000_000])   # a copy: no view may keep the file mapped (Windows lock)
            del mm

    # ------------------------------------------------------------------ val / train split
    # rows 0..nL-1 = labelled, nL.. = rival context. X holds labelled rows only (index it with mask[:nL]).
    u1, r1 = np.unique(np.concatenate([S1ID, CS1]), return_inverse=True)
    uo, ro = np.unique(np.concatenate([OID, CO]),   return_inverse=True)
    del S1ID, OID, CS1, CO
    labm = np.r_[np.ones(nL, dtype=bool), np.zeros(nC, dtype=bool)]
    Y = np.r_[Y, np.zeros(nC, dtype=np.int8)]   # context labels are never used for training/scoring

    if a.final:
        # No holdout — every labelled pair is used for training.
        tr     = labm.copy()
        is_val = np.zeros(len(Y), dtype=bool)
        print("--final mode: training on ALL data (no validation holdout)")
    else:
        is_val = np.array([zlib.crc32((s + "v").encode()) % 10 < 3 for s in u1])[r1] & labm
        tr     = ~is_val & labm
        if a.train_frac < 1.0:
            sub = np.array([hashlib.md5((s + "t").encode()).digest()[0] < 256 * a.train_frac for s in u1])[r1]
            tr = tr & sub
            print(f"--train-frac {a.train_frac}: training on {int(tr.sum()):,} pairs "
                  f"({len(np.unique(r1[tr])):,} S1) - val unchanged", flush=True)
    # Val is split per S1 entity into two halves:
    #   ES  half -> early stopping (tree count) + thr/margin choice  (used for decisions)
    #   REP half -> never used for any choice: an honest, unbiased report score
    es_half = np.array([es_half_of(s) for s in u1])[r1]
    is_es   = is_val & es_half
    is_rep  = is_val & ~es_half

    fold = np.array([zlib.crc32((s + "f").encode()) % 2 for s in u1])[r1]

    # ------------------------------------------------------------------ stage 1
    # Out-of-fold p1 on training portion; direct p1 on val portion.
    p1 = np.zeros(len(Y), dtype=np.float32)
    L = lambda m: m[:nL]   # mask over all rows -> mask over X's labelled rows
    if a.final:
        # Only one real fold (all data is training); do a single 50/50 internal OOF for p1 quality,
        # then refit on everything for the final stage-1 model.
        for f in (0, 1):
            fit_mask  = (fold == f) & labm
            pred_mask = (fold != f) & labm   # predict on the OTHER half
            # Guard: if all positives end up in one fold, skip
            if Y[fit_mask].sum() == 0 or Y[pred_mask].sum() == 0:
                p1[pred_mask] = 0.5
            else:
                p1[pred_mask] = fit_stage1(X[L(fit_mask)], Y[fit_mask]).predict_proba(X[L(pred_mask)])[:, 1]
        m1 = fit_stage1(X, Y[labm])      # final stage-1 on ALL labelled data
    else:
        # main model first: early stopping on the ES half picks the tree count n1 ...
        m1 = fit_stage1(X[L(tr)], Y[tr], eval_set=(X[L(is_es)], Y[is_es]))
        n1 = m1.best_iteration_ or m1.n_estimators
        print(f"stage-1 trees: {n1} (cap {m1.n_estimators})", flush=True)
        # ... then the OOF fold models use the same n1, so train-p1 and val-p1 come from equal-size models
        for f in (0, 1):
            fit  = tr & (fold != f)
            pred = tr & (fold == f)
            p1[pred] = fit_stage1(X[L(fit)], Y[fit], n_estimators=n1).predict_proba(X[L(pred)])[:, 1]
        p1[is_val] = m1.predict_proba(X[L(is_val)])[:, 1]
        rest = ~tr & ~is_val & labm   # only non-empty with --train-frac < 1: scored like test rows
        if rest.any():
            p1[rest] = m1.predict_proba(X[L(rest)])[:, 1]
    # rival-context rows: scored by the final stage-1 model exactly like test rows; keep their RAW2 cols
    raw_parts = [raw2(X)]
    if nC:
        s = nL
        for blk in ctx_chunks():
            p1[s:s + len(blk)] = m1.predict_proba(blk)[:, 1]
            raw_parts.append(raw2(blk)); s += len(blk)
        del blk
        print(f"rival context scored: {nC:,} pairs", flush=True)

    # ------------------------------------------------------------------ stage 2
    X2 = stage2_matrix(r1, ro, p1, np.vstack(raw_parts)); del raw_parts
    if a.final:
        m2 = fit_stage2(X2[labm], Y[labm])
    else:
        m2 = fit_stage2(X2[tr], Y[tr], eval_set=(X2[is_es], Y[is_es]))
        print(f"stage-2 trees: {m2.best_iteration_ or m2.n_estimators} (cap {m2.n_estimators})", flush=True)
    p2 = m2.predict_proba(X2)[:, 1]

    # ------------------------------------------------------------------ threshold / margin sweep
    if a.final:
        # thr/margin already locked — no sweep needed.
        best = (None, float(a.thr), float(a.margin))
        print(f"--final mode: using fixed thr={best[1]:.2f} margin={best[2]:.2f} (no sweep)")

    elif a.thr is not None and a.margin is not None:
        # Caller supplied fixed values — skip the sweep (useful for quick re-trains).
        best = (None, float(a.thr), float(a.margin))
        print(f"fixed thr/margin supplied: thr={best[1]:.2f} margin={best[2]:.2f} (sweep skipped)")

    else:
        # ---- Fine-grained sweep ----
        # WIDENED again (26 Sep, post is_unbalance=True): a prior run with is_unbalance=True hit
        # thr=0.91 margin=0.30, right at the old grid's edge (thr 0.45-0.92, margin 0.00-0.32) -
        # that's a sign the true optimum is outside the tested range, not that 0.91/0.30 is it.
        # is_unbalance reweights the loss and pushes predicted probabilities more extreme, which
        # shifts where the best thr/margin sits - so widen until the winner stops landing on an edge.
        all_val = {k for k in truth if zlib.crc32((k + "v").encode()) % 10 < 3}
        val_ids = {k for k in all_val if es_half_of(k)}                              # ES half: choose here
        rep_ids = all_val - val_ids                                                  # REP half: report only
        vt      = {k: v for k, v in truth.items() if k in val_ids}
        best    = (-1.0, 0.5, 0.0)

        # 27 Sep: extended past 0.98 - at realistic density (sample_v2) the optimum landed exactly on 0.98,
        # the old grid's top edge, so the true optimum may be stricter.
        thr_grid    = np.concatenate([np.arange(0.30, 0.98, 0.02), [0.98, 0.985, 0.99, 0.993, 0.996, 0.998]])
        margin_grid = np.arange(0.00, 0.45, 0.02)
        print(f"sweeping {len(thr_grid)} thresholds x {len(margin_grid)} margins "
              f"= {len(thr_grid)*len(margin_grid)} combos …")

        # decode_prep does the expensive sort/groupby ONCE (thr/margin don't affect it);
        # decode_apply per combo is then just a cheap boolean filter. The old code called
        # decode() (full re-sort) 384 times - this is the same result, much faster.
        prepped = decode_prep(r1, ro, p2)

        for thr in thr_grid:
            for mg in margin_grid:
                rr, oo = decode_apply(prepped, thr, mg)
                pred = {}
                for i, o in zip(rr, oo):
                    s = u1[i]
                    if s in val_ids:
                        pred.setdefault(s, set()).add(uo[o])
                sc = f05_macro(pred, vt)
                if sc > best[0]:
                    best = (sc, float(thr), float(mg))

        print(f"ES-half macro F0.5 = {best[0]:.4f}  "
              f"at thr={best[1]:.3f} margin={best[2]:.2f}  (ES S1: {len(vt)})  [used for choices]")

        # Honest scores at the chosen thr/margin. REP half was never used for any choice.
        rr, oo = decode_apply(prepped, best[1], best[2])
        pred_all = {}
        for i, o in zip(rr, oo):
            s = u1[i]
            if s in all_val:
                pred_all.setdefault(s, set()).add(uo[o])
        rt = {k: v for k, v in truth.items() if k in rep_ids}
        at = {k: v for k, v in truth.items() if k in all_val}
        rep_f05 = f05_macro({k: v for k, v in pred_all.items() if k in rep_ids}, rt)
        all_f05 = f05_macro(pred_all, at)
        print(f"CLEAN report-half macro F0.5 = {rep_f05:.4f}  (REP S1: {len(rt)})  <- honest number")
        print(f"full-val macro F0.5          = {all_f05:.4f}  (val S1: {len(at)})  <- compare to v7 0.9452")

        # ---- expected-F0.5 decoder: fitted on the ES half only, kept only if it beats thr on ES
        ra, oa, pa = best_assignment(r1, ro, p2)
        nkey = int(ro.max()) + 1
        ks = np.sort(r1.astype(np.int64) * nkey + ro)
        yk = Y[np.argsort(r1.astype(np.int64) * nkey + ro)]
        ya = yk[np.searchsorted(ks, ra * nkey + oa)].astype(bool)
        s_of = u1[ra]
        es_pair = np.fromiter((s in val_ids for s in s_of), dtype=bool, count=len(ra))
        es_s1 = sorted(val_ids)
        found = pd.Series(ya[es_pair]).groupby(s_of[es_pair]).sum()
        dec = fit_ef_decoder(pa[es_pair], ya[es_pair], [len(truth[s]) for s in es_s1],
                             [int(found.get(s, 0)) for s in es_s1])
        keep = decode_ef_assigned(ra, pa, dec)
        pred_ef = {}
        for i, o in zip(ra[keep], oa[keep]):
            s = u1[i]
            if s in all_val:
                pred_ef.setdefault(s, set()).add(uo[o])
        ef_es = f05_macro({k: v for k, v in pred_ef.items() if k in val_ids}, vt)
        ef_rep = f05_macro({k: v for k, v in pred_ef.items() if k in rep_ids}, rt)
        ef_all = f05_macro(pred_ef, at)
        print(f"EF decoder (lam={dec['lam']:.3f}): ES {ef_es:.4f} | CLEAN {ef_rep:.4f} | full-val {ef_all:.4f}")
        use_ef = ef_es > best[0]
        print(f"-> {'EF decoder beats thr on ES: saving decoder.json' if use_ef else 'thr decoder kept (EF not better on ES)'}")

    # ------------------------------------------------------------------ save
    m1.booster_.save_model(f"{a.models}/stage1.txt")
    m2.booster_.save_model(f"{a.models}/stage2.txt")
    val_f05 = best[0] if (best[0] is not None) else "N/A (final mode)"
    extra = {}
    if not a.final and best[0] is not None:
        extra = dict(rep_f05_clean=rep_f05, full_val_f05=all_f05,
                     n_trees_stage1=int(m1.best_iteration_ or m1.n_estimators),
                     n_trees_stage2=int(m2.best_iteration_ or m2.n_estimators),
                     ef_es_f05=ef_es, ef_rep_f05_clean=ef_rep, ef_full_val_f05=ef_all, ef_used=bool(use_ef))
        if use_ef:
            dec.update(es_f05=ef_es, rep_f05=ef_rep, thr_es_f05=best[0], thr_rep_f05=rep_f05)
            json.dump(dec, open(f"{a.models}/decoder.json", "w"), indent=1)
        elif os.path.exists(f"{a.models}/decoder.json"):
            os.remove(f"{a.models}/decoder.json")   # never leave a stale decoder next to new models
    json.dump(
        dict(cfg=CFG, thr=best[1], margin=best[2], val_f05=val_f05,
             final_mode=a.final, **extra),
        open(f"{a.models}/config.json", "w"), indent=2
    )
    print(f"saved models to {a.models}/")
    if C_FILES:
        import shutil
        shutil.rmtree(ctx_dir, ignore_errors=True)   # temporary rival-context feature files
    if a.final:
        print("NOTE: these models were trained on ALL data. "
              "The thr/margin were locked before training — do NOT retune them on this run.")


if __name__ == "__main__":
    main()
