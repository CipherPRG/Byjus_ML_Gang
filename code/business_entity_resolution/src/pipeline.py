"""Shared per-country processing: load -> normalise -> block -> features."""
import os
import numpy as np
import pandas as pd
from blocking import Side, candidates
from features import pair_features
from io_utils import read_tsv, read_country
from norm import generic_addr_tokens, learn_legal_sk, norm_addr, norm_name, skel


def load_side(path, country, chunksize=300_000, stop=frozenset(), kv2=False, extra_legal_sk=frozenset()):
    sides = []
    for ch in read_tsv(path, chunksize=chunksize):
        ch = ch[ch.country == country]
        if len(ch):
            sides.append(Side(ch, stop, kv2, extra_legal_sk))
    return Side.concat(sides) if sides else None


def country_stop(s1_df, cfg):
    """Per-country generic address words, learned from that country's Source 1 addresses.
    Controlled by cfg['addr_stop_frac'] (fraction of addresses a word must appear in);
    missing/0 = off, so configs of models trained before this existed reproduce exactly."""
    frac = cfg.get("addr_stop_frac", 0.0)
    if not frac:
        return frozenset()
    return generic_addr_tokens(s1_df.business_address.values, frac=frac)


def build_country(dir_, prefix, country, cfg, workers=1, density_src=None):
    """Returns (s1, oth, cand) for one country. `prefix` = 'train' or 'test'.
    feat_v3 (cfg): also attaches oth.dens_addr / oth.dens_name = how many S1 businesses (per 100k S1
    of the country) share this record's normalised address / name skeleton, counted in the FULL S1
    file `density_src` (default: this folder's source1; pass the full train_source1 when training on
    a sample, which has ~20x fewer S1). learn_suffix (cfg): legal suffixes learned from this S1 file."""
    s1_df = read_country(f"{dir_}/{prefix}_source1.tsv", country)
    stop = country_stop(s1_df, cfg)
    kv2 = bool(cfg.get("keys_v2", False))  # missing in older configs -> exact old behaviour
    xsk = learn_legal_sk(s1_df.business_name.values) if cfg.get("learn_suffix") else frozenset()
    s1 = Side(s1_df, stop, kv2, xsk)
    del s1_df
    parts = [load_side(f"{dir_}/{prefix}_source{k}.tsv", country, stop=stop, kv2=kv2, extra_legal_sk=xsk)
             for k in (2, 3)]
    parts = [p for p in parts if p is not None]
    if not parts:
        return s1, None, None
    oth = Side.concat(parts)
    if cfg.get("feat_v3"):
        attach_density(oth, density_src or f"{dir_}/{prefix}_source1.tsv", country, kv2, xsk)
    pf = f"{dir_}/{prefix}_pairs.tsv"
    if os.path.exists(pf):
        # sample_v3: use the REAL full-density candidate pairs recorded by make_sample_v3.py instead of
        # re-blocking inside the sample (which would give the wrong rival counts). Never present for test.
        cand = load_pairs(pf, country, s1, oth)
    else:
        cand = candidates(s1, oth, max_block=cfg["max_block"], max_s1_block=cfg["max_s1_block"], topk=cfg["topk"])
    return s1, oth, cand


def attach_density(oth, s1_path, country, kv2, xsk, chunksize=300_000):
    """feat_v3: per S2/S3 record, how many S1 businesses of this country (in the FULL S1 file) share its
    exact normalised address / name skeleton, scaled per 100k S1 so train and test sizes compare.
    High = non-specific (sibling businesses at one address, very common name)."""
    from collections import Counter
    ca, cn, n = Counter(), Counter(), 0
    for ch in read_tsv(s1_path, chunksize=chunksize):
        ch = ch[ch.country == country]
        for nm, ad in zip(ch.business_name.values, ch.business_address.values):
            ca[norm_addr(ad, ords=kv2)[0]] += 1
            _, core = norm_name(nm, xsk)
            cn[" ".join(x for x in (skel(t) for t in core) if x)] += 1
            n += 1
    k = 1e5 / max(n, 1)
    oth.dens_addr = np.fromiter((ca.get(a, 0) * k for a in oth.aclean), dtype=np.float32, count=len(oth))
    oth.dens_name = np.fromiter((cn.get(s, 0) * k for s in oth.nskel), dtype=np.float32, count=len(oth))
    print(f"  [{country}] feat_v3 density from {os.path.basename(s1_path)}: {n:,} S1", flush=True)


def load_pairs(path, country, s1, oth, chunksize=2_000_000):
    i1 = {e: i for i, e in enumerate(s1.ids)}
    io = {e: i for i, e in enumerate(oth.ids)}
    R1, RO, W = [], [], []
    for ch in pd.read_csv(path, sep="\t", dtype={"s1_id": str, "oth_id": str, "w": np.float32, "country": str},
                          chunksize=chunksize, keep_default_na=False):
        ch = ch[ch.country == country]
        if not len(ch):
            continue
        a = ch.s1_id.map(i1); b = ch.oth_id.map(io)
        ok = a.notna() & b.notna()
        R1.append(a[ok].values.astype(np.int32)); RO.append(b[ok].values.astype(np.int32))
        W.append(ch.w[ok].values.astype(np.float32))
    if not R1:
        return pd.DataFrame({"r1": np.array([], np.int32), "ro": np.array([], np.int32), "w": np.array([], np.float32)})
    out = pd.DataFrame({"r1": np.concatenate(R1), "ro": np.concatenate(RO), "w": np.concatenate(W)})
    print(f"  [{country}] using recorded candidate pairs from {os.path.basename(path)}: {len(out):,}", flush=True)
    return out.sort_values(["r1", "w"], ascending=[True, False], kind="stable").reset_index(drop=True)
