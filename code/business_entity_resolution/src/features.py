"""Pair features (computed in worker processes from plain string lists)."""
import re
from functools import lru_cache
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from norm import LEGAL, addr_keys, _hasdig

# Base feature list (36 columns). The order is part of the saved model: never reorder, only append.
F1 = ["nr", "nsort", "nset", "npart", "njw", "nlev", "ncore_eq", "ncore_jac", "nlen_d", "nfirst_eq", "ncomp",
      "ar", "asort", "aset", "apart", "ajac", "anum_jac", "anum_eq", "a2_empty",
      "akey_eq", "a_exact", "anum_first_eq", "sk_r", "sk_set", "sk_part", "w",
      "ajw", "alev", "alen_d", "ncontain", "akey_jac", "wcount_d",
      "a1_empty", "both_empty",
      "hnum_edit", "hnum_logdiff"]

# Extended feature list (42 columns, feat_v3=True): six features appended at the end so that
# F1[:36] == F1_V3[:36] and existing RAW2 index lookups are unaffected.
F1_V3 = F1 + [
    "nspan_jac",  # Jaccard over pure digit runs in each address; more robust than anum_jac for
                  # mixed-format numbers like "12-A" vs "12" (anum_jac=0, nspan_jac=1.0)
    "sk_eq",      # exact match of the full sorted skeleton string; tighter than sk_r fuzz ratio
    # IDF-weighted similarity (weights = rarity of the word among the country's S1 records, learned
    # from the S1 file itself, no labels): a shared rare word is strong evidence, a shared common word
    # ('private', 'road', city names) is weak, a gibberish name shares nothing.
    "n_idf_cos", "n_idf_max", "a_idf_cos", "a_idf_max",
]

# IDF tables for the worker processes (set once per pool via _init_idf; empty -> features are 0)
_IDF = {"n": {}, "a": {}, "mx": 1.0}


def _init_idf(d):
    global _IDF
    _IDF = d


def _wsim(A, B, w, mx, miss):
    """IDF-weighted cosine of two token sets + the (normalised) weight of the rarest shared token."""
    if not A or not B:
        return 0.0, 0.0
    wa = {t: w.get(t, miss) for t in A}; wb = {t: w.get(t, miss) for t in B}
    sh = A & B
    if not sh:
        return 0.0, 0.0
    num = sum(wa[t] * wa[t] for t in sh)
    den = (sum(v * v for v in wa.values()) * sum(v * v for v in wb.values())) ** 0.5
    return (num / den if den else 0.0), max(wa[t] for t in sh) / mx


def _core(n):
    t = n.split()
    return [x for x in t if x not in LEGAL] or t


def _jac(a, b):
    if not a or not b:
        return 0.0
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)




def _hnum(t1, t2):
    """S1 house number vs the CLOSEST number token anywhere in the other address.
    Separates a typo'd house number (1400 -> 1402, 407 -> 07: edit distance 1) from a genuinely
    different building (46 -> 53: edit distance 2+). On sample data, among candidates with
    near-identical names, ~87% (US) / ~61% (India) of wrong pairs are edit 2+, vs ~5% of true pairs.
    Returns (edit distance capped at 3, log1p numeric gap); (-1, -1) if either side has no number."""
    n1 = [t for t in t1 if _hasdig(t)]
    n2 = [t for t in t2 if _hasdig(t)]
    if not n1 or not n2:
        return -1.0, -1.0
    h = n1[0]
    best = min(n2, key=lambda y: Levenshtein.distance(h, y))
    e = min(Levenshtein.distance(h, best), 3)
    a = "".join(ch for ch in h if ch.isdigit())[:9]
    b = "".join(ch for ch in best if ch.isdigit())[:9]
    gap = float(np.log1p(abs(int(a) - int(b)))) if a and b else -1.0
    return float(e), gap


@lru_cache(maxsize=1 << 16)
def _pname(n):
    """Per-record name parts (pure function of the string; cached because every record appears in
    ~20 candidate pairs, and the S1 side is consecutive since candidates are sorted by r1)."""
    c = _core(n)
    return c, " ".join(c)


@lru_cache(maxsize=1 << 16)
def _paddr(a, stop):
    """Per-record address parts: tokens, number tokens (in order), their set, blocking keys, first number."""
    t = a.split()
    d = [x for x in t if _hasdig(x)]
    return t, d, set(d), set(addr_keys(t, stop=stop)), (d[0] if d else None)


def _hnum_d(n1, n2):
    """_hnum() on precomputed number-token lists (same result, no re-scan of the tokens)."""
    if not n1 or not n2:
        return -1.0, -1.0
    h = n1[0]
    best = min(n2, key=lambda y: Levenshtein.distance(h, y))
    e = min(Levenshtein.distance(h, best), 3)
    a = "".join(ch for ch in h if ch.isdigit())[:9]
    b = "".join(ch for ch in best if ch.isdigit())[:9]
    gap = float(np.log1p(abs(int(a) - int(b)))) if a and b else -1.0
    return float(e), gap


def _chunk(args):
    n1s, a1s, n2s, a2s, k1s, k2s, ws, stop, feat_v3 = args
    ncols = len(F1_V3) if feat_v3 else len(F1)
    out = np.zeros((len(n1s), ncols), dtype=np.float32)
    for i, (n1, a1, n2, a2, sk1, sk2, w) in enumerate(zip(n1s, a1s, n2s, a2s, k1s, k2s, ws)):
        c1, j1 = _pname(n1); c2, j2 = _pname(n2)
        t1, d1, nu1, k1, f1 = _paddr(a1, stop); t2, d2, nu2, k2, f2 = _paddr(a2, stop)
        e1, e2 = not a1, not a2
        both = not (e1 or e2)
        h_edit, h_gap = _hnum_d(d1, d2)
        # base 36 features (F1 order)
        base = (
            fuzz.ratio(n1, n2), fuzz.token_sort_ratio(n1, n2), fuzz.token_set_ratio(n1, n2), fuzz.partial_ratio(n1, n2),
            JaroWinkler.similarity(j1, j2), Levenshtein.normalized_similarity(j1, j2),
            float(sorted(c1) == sorted(c2)), _jac(c1, c2), abs(len(j1) - len(j2)) / max(len(j1), len(j2), 1),
            float(bool(c1) and bool(c2) and c1[0] == c2[0]), fuzz.token_set_ratio(j1, j2),
            fuzz.ratio(a1, a2) if both else 0.0, fuzz.token_sort_ratio(a1, a2) if both else 0.0,
            fuzz.token_set_ratio(a1, a2) if both else 0.0, fuzz.partial_ratio(a1, a2) if both else 0.0,
            _jac(t1, t2), _jac(nu1, nu2), float(bool(nu1 & nu2)), float(e2),
            float(bool(k1 & k2)), float(both and a1 == a2), float(f1 is not None and f1 == f2),
            fuzz.ratio(sk1, sk2), fuzz.token_set_ratio(sk1, sk2), fuzz.partial_ratio(sk1, sk2), float(w),
            JaroWinkler.similarity(a1, a2) if both else 0.0, Levenshtein.normalized_similarity(a1, a2) if both else 0.0,
            abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1) if both else 0.0,
            float(len(j1) >= 4 and len(j2) >= 4 and (j1 in j2 or j2 in j1)),
            _jac(k1, k2), abs(len(c1) - len(c2)) / max(len(c1), len(c2), 1),
            float(e1), float(e1 and e2),
            h_edit, h_gap)
        if feat_v3:
            # Two extra features appended — only computed / stored when feat_v3=True
            sp1 = set(re.findall(r'\d+', a1)); sp2 = set(re.findall(r'\d+', a2))
            nspan_jac = _jac(sp1, sp2)
            sk_eq = float(bool(sk1) and bool(sk2) and sk1 == sk2)
            mx, miss = _IDF["mx"], _IDF.get("miss", _IDF["mx"])
            nc, nm = _wsim(set(c1), set(c2), _IDF["n"], mx, miss)
            ac, am = _wsim(set(t1), set(t2), _IDF["a"], mx, miss) if both else (0.0, 0.0)
            out[i] = base + (nspan_jac, sk_eq, nc, nm, ac, am)
        else:
            out[i] = base
    return out


def pair_features(s1, oth, r1, ro, w, workers=1, chunk=20000, feat_v3=False):
    """Feature matrix float32 for pairs (S1 row r1, other row ro).
    feat_v3=False (default): shape (N, 36) = F1.
    feat_v3=True:            shape (N, 42) = F1_V3 (F1 + digit-span Jaccard, skeleton equality, IDF similarities)."""
    stop = getattr(s1, "stop", frozenset())  # per-country generic address words, same set blocking used
    jobs = []
    for s in range(0, len(r1), chunk):
        a, b = r1[s:s + chunk], ro[s:s + chunk]
        jobs.append(([s1.nclean[i] for i in a], [s1.aclean[i] for i in a],
                     [oth.nclean[i] for i in b], [oth.aclean[i] for i in b],
                     [s1.nskel[i] for i in a], [oth.nskel[i] for i in b], w[s:s + chunk], stop, feat_v3))
    idf = getattr(s1, "idf", None) if feat_v3 else None   # set by pipeline.attach_density (feat_v3)
    if feat_v3 and idf is None:
        idf = {"n": {}, "a": {}, "mx": 1.0}
    if workers > 1 and len(jobs) > 1:
        kw = dict(initializer=_init_idf, initargs=(idf,)) if feat_v3 else {}
        with ProcessPoolExecutor(workers, **kw) as ex:
            res = list(ex.map(_chunk, jobs))
    else:
        if feat_v3:
            _init_idf(idf)
        res = [_chunk(j) for j in jobs]
    ncols = len(F1_V3) if feat_v3 else len(F1)
    return np.vstack(res) if res else np.zeros((0, ncols), dtype=np.float32)
