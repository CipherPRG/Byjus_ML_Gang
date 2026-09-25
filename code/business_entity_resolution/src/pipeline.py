"""Shared per-country processing: load -> normalise -> block -> features."""
import numpy as np
import pandas as pd
from blocking import Side, candidates
from features import pair_features
from io_utils import read_tsv, read_country


def load_side(path, country, chunksize=300_000):
    sides = []
    for ch in read_tsv(path, chunksize=chunksize):
        ch = ch[ch.country == country]
        if len(ch):
            sides.append(Side(ch))
    return Side.concat(sides) if sides else None


def build_country(dir_, prefix, country, cfg, workers=1):
    """Returns (s1, oth, cand) for one country. `prefix` = 'train' or 'test'."""
    s1 = Side(read_country(f"{dir_}/{prefix}_source1.tsv", country))
    parts = [load_side(f"{dir_}/{prefix}_source{k}.tsv", country) for k in (2, 3)]
    parts = [p for p in parts if p is not None]
    if not parts:
        return s1, None, None
    oth = Side.concat(parts)
    cand = candidates(s1, oth, max_block=cfg["max_block"], max_s1_block=cfg["max_s1_block"], topk=cfg["topk"])
    return s1, oth, cand
