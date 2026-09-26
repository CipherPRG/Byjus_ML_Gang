"""Shared per-country processing: load -> normalise -> block -> features."""
import numpy as np
import pandas as pd
from blocking import Side, candidates
from features import pair_features
from io_utils import read_tsv, read_country
from norm import generic_addr_tokens


def load_side(path, country, chunksize=300_000, stop=frozenset()):
    sides = []
    for ch in read_tsv(path, chunksize=chunksize):
        ch = ch[ch.country == country]
        if len(ch):
            sides.append(Side(ch, stop))
    return Side.concat(sides) if sides else None


def country_stop(s1_df, cfg):
    """Per-country generic address words, learned from that country's Source 1 addresses.
    Controlled by cfg['addr_stop_frac'] (fraction of addresses a word must appear in);
    missing/0 = off, so configs of models trained before this existed reproduce exactly."""
    frac = cfg.get("addr_stop_frac", 0.0)
    if not frac:
        return frozenset()
    return generic_addr_tokens(s1_df.business_address.values, frac=frac)


def build_country(dir_, prefix, country, cfg, workers=1):
    """Returns (s1, oth, cand) for one country. `prefix` = 'train' or 'test'."""
    s1_df = read_country(f"{dir_}/{prefix}_source1.tsv", country)
    stop = country_stop(s1_df, cfg)
    s1 = Side(s1_df, stop)
    del s1_df
    parts = [load_side(f"{dir_}/{prefix}_source{k}.tsv", country, stop=stop) for k in (2, 3)]
    parts = [p for p in parts if p is not None]
    if not parts:
        return s1, None, None
    oth = Side.concat(parts)
    cand = candidates(s1, oth, max_block=cfg["max_block"], max_s1_block=cfg["max_s1_block"], topk=cfg["topk"])
    return s1, oth, cand
