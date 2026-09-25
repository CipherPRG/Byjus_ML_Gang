"""Pair features (computed in worker processes from plain string lists)."""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from norm import LEGAL, addr_keys

F1 = ["nr", "nsort", "nset", "npart", "njw", "nlev", "ncore_eq", "ncore_jac", "nlen_d", "nfirst_eq", "ncomp",
      "ar", "asort", "aset", "apart", "ajac", "anum_jac", "anum_eq", "a1_empty", "a2_empty",
      "akey_eq", "a_exact", "anum_first_eq", "sk_r", "sk_set", "sk_part", "w"]


def _core(n):
    t = n.split()
    return [x for x in t if x not in LEGAL] or t


def _jac(a, b):
    if not a or not b:
        return 0.0
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def _hasdig(t):
    return any(ch.isdigit() for ch in t)


def _chunk(args):
    n1s, a1s, n2s, a2s, k1s, k2s, ws = args
    out = np.zeros((len(n1s), len(F1)), dtype=np.float32)
    for i, (n1, a1, n2, a2, sk1, sk2, w) in enumerate(zip(n1s, a1s, n2s, a2s, k1s, k2s, ws)):
        c1, c2 = _core(n1), _core(n2)
        j1, j2 = " ".join(c1), " ".join(c2)
        t1, t2 = a1.split(), a2.split()
        nu1 = {t for t in t1 if _hasdig(t)}; nu2 = {t for t in t2 if _hasdig(t)}
        e1, e2 = not a1, not a2
        both = not (e1 or e2)
        k1, k2 = set(addr_keys(t1)), set(addr_keys(t2))
        f1 = next((t for t in t1 if _hasdig(t)), None); f2 = next((t for t in t2 if _hasdig(t)), None)
        out[i] = (
            fuzz.ratio(n1, n2), fuzz.token_sort_ratio(n1, n2), fuzz.token_set_ratio(n1, n2), fuzz.partial_ratio(n1, n2),
            JaroWinkler.similarity(j1, j2), Levenshtein.normalized_similarity(j1, j2),
            float(sorted(c1) == sorted(c2)), _jac(c1, c2), abs(len(j1) - len(j2)) / max(len(j1), len(j2), 1),
            float(bool(c1) and bool(c2) and c1[0] == c2[0]), fuzz.token_set_ratio(j1, j2),
            fuzz.ratio(a1, a2) if both else 0.0, fuzz.token_sort_ratio(a1, a2) if both else 0.0,
            fuzz.token_set_ratio(a1, a2) if both else 0.0, fuzz.partial_ratio(a1, a2) if both else 0.0,
            _jac(t1, t2), _jac(nu1, nu2), float(bool(nu1 & nu2)), float(e1), float(e2),
            float(bool(k1 & k2)), float(both and a1 == a2), float(f1 is not None and f1 == f2),
            fuzz.ratio(sk1, sk2), fuzz.token_set_ratio(sk1, sk2), fuzz.partial_ratio(sk1, sk2), float(w))
    return out


def pair_features(s1, oth, r1, ro, w, workers=1, chunk=20000):
    """Feature matrix (len(r1) x len(F1)) float32 for pairs (S1 row r1, other row ro)."""
    jobs = []
    for s in range(0, len(r1), chunk):
        a, b = r1[s:s + chunk], ro[s:s + chunk]
        jobs.append(([s1.nclean[i] for i in a], [s1.aclean[i] for i in a],
                     [oth.nclean[i] for i in b], [oth.aclean[i] for i in b],
                     [s1.nskel[i] for i in a], [oth.nskel[i] for i in b], w[s:s + chunk]))
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_chunk, jobs))
    else:
        res = [_chunk(j) for j in jobs]
    return np.vstack(res) if res else np.zeros((0, len(F1)), dtype=np.float32)
