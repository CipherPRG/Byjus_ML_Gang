"""Blocking / candidate generation. Country-scoped keys, numpy/pandas merges (memory-lean)."""
import zlib
import numpy as np
import pandas as pd
from norm import norm_name, norm_addr, addr_keys, ADDR_STOP, skel, _hasdig

# weight of each key type (address keys are the most reliable)
WEIGHT = {"A": 3, "R": 2, "N": 3, "Q": 2, "P": 2, "T": 1, "U": 1, "S": 1, "K": 1, "V": 2, "X": 3,
          "B": 2, "Y": 2, "C": 3, "M": 2, "L": 2}
_DOM = ("com", "net", "org")




def keys_v2(country, core, atoks, stop):
    """Extra keys (cfg keys_v2=True), each aimed at a blocking-miss pattern seen on sample data:
    B  adjacent address bigram containing a number: '4600 24th', '5534 10480', 'c 25', '14 109'
    Y  unordered (number, rare address word): 'no226|villupuram' even when word order differs
    C  compact name: 'allshivsystems' / initials+last 'wmbrokerage', matching website-style names
       'allshivsystemscom' / 'wmbrokeragecom' (single token ending in com/net/org, suffix stripped)"""
    ks = []
    nb = 0
    for x, y in zip(atoks, atoks[1:]):
        if (_hasdig(x) or _hasdig(y)) and len(x) <= 8 and len(y) <= 8 \
                and x not in ADDR_STOP and y not in ADDR_STOP and x not in stop and y not in stop:
            ks.append("B|" + country + "|" + x + "|" + y); nb += 1
            if nb >= 3:
                break
    nums = [t for t in atoks if _hasdig(t) and len(t) <= 8][:2]
    words = sorted({t for t in atoks if len(t) >= 4 and t.isalpha() and t not in ADDR_STOP and t not in stop},
                   key=lambda t: (-len(t), t))[:4]
    for n in nums:
        for wd in words:
            ks.append("Y|" + country + "|" + n + "|" + wd)
    if len(core) == 1 and len(core[0]) >= 7 and core[0].endswith(_DOM):
        ks.append("C|" + country + "|" + core[0][:-3])
    elif core:
        j = "".join(core)
        if len(j) >= 6:
            ks.append("C|" + country + "|" + j)
        if len(core) >= 2:
            ini = "".join(t[0] for t in core[:-1]) + core[-1]
            if len(ini) >= 6 and ini != j:
                ks.append("C|" + country + "|" + ini)
    return ks


def _pairs(toks):
    return [toks[i] + "|" + toks[j] for i in range(len(toks)) for j in range(i + 1, len(toks))]


def keys_v3(country, core, atoks, stop):
    """EXTRA candidate keys (cfg keys_v3=True, predict/eval time only; kept separate from the normal keys so
    the normal candidates and their weights stay exactly as before). Aimed at the two biggest miss types:
    M  unordered pairs among the 3 longest distinct alphabetic core-name words (len>=4): a word pair is far
       sharper than one word, and 3 pairs survive one extra / missing / typo'd word; works with an EMPTY address.
    L  unordered pairs among the 4 longest distinct non-generic address words (len>=5): survives a typo in
       one address word (single-word R/S keys do not)."""
    ks = []
    nt = sorted({t for t in core if len(t) >= 4 and t.isalpha()}, key=lambda t: (-len(t), t))[:3]
    if len(nt) >= 2:
        ks += ["M|" + country + "|" + x for x in _pairs(sorted(nt))]
    at = sorted({t for t in atoks if len(t) >= 5 and t.isalpha() and t not in ADDR_STOP and t not in stop},
                key=lambda t: (-len(t), t))[:4]
    if len(at) >= 2:
        ks += ["L|" + country + "|" + x for x in _pairs(sorted(at))]
    return ks


def keys_for(country, core, atoks, stop=frozenset(), kv2=False):
    """All normal blocking keys of one record (country-scoped strings; typed by their first letter)."""
    ks = keys_v2(country, core, atoks, stop) if kv2 else []
    for k in addr_keys(atoks, stop=stop):
        ks.append("A|" + country + "|" + k)
    if core:
        ks.append("N|" + country + "|" + " ".join(sorted(core)))
        ks.append("P|" + country + "|" + core[0][:5] + "|" + core[-1][:4])
        if len(core) > 1:
            ks.append("Q|" + country + "|" + "".join(sorted(core))[:9])
        long = sorted({t for t in core if len(t) >= 5 and t.isalpha()}, key=lambda t: (-len(t), t))[:2]
        for t in long:
            ks.append("T|" + country + "|" + t[:4])
            ks.append("U|" + country + "|" + t[-4:])
    sk = [skel(t) for t in core]; sk = [x for x in sk if len(x) >= 2]
    if sk:
        ks.append("V|" + country + "|" + "".join(sorted(sk)))
        for x in sorted(set(sk), key=lambda z: (-len(z), z))[:2]:
            if len(x) >= 3:
                ks.append("K|" + country + "|" + x[:3])
    # compound keys: name-key x address-token. Generic name keys become sharp when paired with an address token.
    nk = []
    if sk:
        nk.append("v" + "".join(sorted(sk)))
        nk += ["k" + x[:3] for x in sorted(set(sk), key=lambda z: (-len(z), z)) if len(x) >= 3][:2]
    lt = sorted({t for t in core if len(t) >= 5 and t.isalpha()}, key=lambda t: (-len(t), t))[:1]
    nk += ["t" + t[:4] for t in lt]
    ak = ["d" + t for t in [t for t in atoks if _hasdig(t) and len(t) <= 8][:2]]
    ak += ["w" + t for t in sorted({t for t in atoks if len(t) >= 5 and t.isalpha() and t not in ADDR_STOP
                                    and t not in stop}, key=lambda t: (-len(t), t))[:2]]
    for a_ in nk[:4]:
        for b_ in ak[:4]:
            ks.append("X|" + country + "|" + a_ + "|" + b_)
    al = sorted({t for t in atoks if len(t) >= 5 and t.isalpha() and t not in ADDR_STOP and t not in stop},
                key=lambda t: (-len(t), t))[:2]
    if len(al) == 2:
        ks.append("R|" + country + "|" + "|".join(sorted(al)))
    if al:
        ks.append("S|" + country + "|" + al[0][:5])
    return ks


def h64(k):
    """Stable 63-bit integer hash of a key string (crc32 << 32 | adler32): identical on every run/OS."""
    b = k.encode("utf-8")
    return ((zlib.crc32(b) & 0x7FFFFFFF) << 32) | zlib.adler32(b)


class Side:
    """Normalised records of one source (one country): ids, clean name/address, blocking keys."""

    def __init__(self, df, stop=frozenset(), kv2=False, extra_legal_sk=frozenset(), kv3=False):
        self.stop = stop  # per-country generic address words (norm.generic_addr_tokens); empty = off
        self.kv2 = kv2    # keys_v2 blocking keys + ordinal normalisation (cfg keys_v2)
        self.extra_legal_sk = extra_legal_sk  # feat_v3: extra learned suffix skeletons; frozenset() = off
        self.ids = df.entity_id.tolist()
        self.nclean, self.aclean, self.nskel = [], [], []
        key, row, w = [], [], []
        key3, row3, w3 = [], [], []   # keys_v3 (extra candidates); stay empty when kv3=False
        for i, (nm, ad, c) in enumerate(zip(df.business_name.values, df.business_address.values, df.country.values)):
            nc, core = norm_name(nm, extra_legal_sk)
            ac, at = norm_addr(ad, ords=kv2)
            self.nclean.append(nc)
            self.nskel.append(" ".join(x for x in (skel(t) for t in core) if x))
            self.aclean.append(ac)
            for k in keys_for(c, core, at, stop, kv2):
                key.append(h64(k)); row.append(i); w.append(WEIGHT[k[0]])
            if kv3:
                for k in keys_v3(c, core, at, stop):
                    key3.append(h64(k)); row3.append(i); w3.append(WEIGHT[k[0]])
        self.key = np.array(key, dtype=np.int64)
        self.krow = np.array(row, dtype=np.int32)
        self.kw = np.array(w, dtype=np.int8)
        self.key3 = np.array(key3, dtype=np.int64)
        self.krow3 = np.array(row3, dtype=np.int32)
        self.kw3 = np.array(w3, dtype=np.int8)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def concat(sides):
        """Merge several Sides (e.g. Source 2 and Source 3 chunks) into one, re-basing the row indices."""
        out = Side.__new__(Side)
        out.stop = sides[0].stop if sides else frozenset()
        out.kv2 = sides[0].kv2 if sides else False
        out.extra_legal_sk = sides[0].extra_legal_sk if sides else frozenset()
        out.ids, out.nclean, out.aclean, out.nskel = [], [], [], []
        keys, rows, ws = [], [], []
        k3, r3, w3 = [], [], []
        off = 0
        for s in sides:
            out.ids += s.ids; out.nclean += s.nclean; out.aclean += s.aclean; out.nskel += s.nskel
            keys.append(s.key); rows.append(s.krow + off); ws.append(s.kw)
            k3.append(s.key3); r3.append(s.krow3 + off); w3.append(s.kw3); off += len(s)
        out.key = np.concatenate(keys); out.krow = np.concatenate(rows); out.kw = np.concatenate(ws)
        out.key3 = np.concatenate(k3); out.krow3 = np.concatenate(r3); out.kw3 = np.concatenate(w3)
        return out


def candidates(s1, oth, max_block=30, max_s1_block=200, topk=25, batch=50_000):
    """Return DataFrame(r1, ro, w): candidate pairs (row indices) and the summed weight of shared keys."""
    k2 = pd.DataFrame({"key": oth.key, "ro": oth.krow, "w": oth.kw})
    k2 = k2[k2.groupby("key").key.transform("size") <= max_block]
    k1 = pd.DataFrame({"key": s1.key, "r1": s1.krow})
    k1 = k1[k1.groupby("key").key.transform("size") <= max_s1_block]
    k1 = k1[k1.key.isin(k2.key.unique())]
    k2 = k2.drop_duplicates(["key", "ro"])
    parts = []
    for start in range(0, len(s1), batch):
        kb = k1[(k1.r1 >= start) & (k1.r1 < start + batch)]
        if kb.empty:
            continue
        m = kb.merge(k2, on="key")
        g = m.groupby(["r1", "ro"], sort=False).w.sum().reset_index()
        g = g.sort_values(["r1", "w"], ascending=[True, False])
        g = g.groupby("r1", sort=False).head(topk)
        parts.append(g)
    if not parts:
        return pd.DataFrame({"r1": np.array([], dtype=np.int64), "ro": np.array([], dtype=np.int64), "w": np.array([])})
    out = pd.concat(parts, ignore_index=True)
    out["r1"] = out.r1.astype(np.int32); out["ro"] = out.ro.astype(np.int32); out["w"] = out.w.astype(np.float32)
    return out


def candidates_extra(s1, oth, base, max_block=60, max_s1_block=200, topk=10, batch=50_000):
    """keys_v3: EXTRA candidate pairs found only by the keys_v3 keys (same block caps as candidates()), minus
    every pair already in `base` (the normal candidates, left exactly as they are). Keeps the `topk` strongest
    extra pairs per S1 by their keys_v3 weight. Returns DataFrame(r1, ro, w) like candidates()."""
    empty = pd.DataFrame({"r1": np.array([], np.int32), "ro": np.array([], np.int32), "w": np.array([], np.float32)})
    if len(s1.key3) == 0 or len(oth.key3) == 0:
        return empty
    ext = candidates(_KeyView(s1.key3, s1.krow3, None, len(s1)), _KeyView(oth.key3, oth.krow3, oth.kw3, len(oth)),
                     max_block=max_block, max_s1_block=max_s1_block, topk=topk + 200, batch=batch)
    if not len(ext):
        return empty
    nk = np.int64(len(oth) + 1)
    have = np.unique(base.r1.values.astype(np.int64) * nk + base.ro.values.astype(np.int64))
    k = ext.r1.values.astype(np.int64) * nk + ext.ro.values.astype(np.int64)
    pos = np.searchsorted(have, k); pos[pos >= len(have)] = 0
    new = ext[~(have[pos] == k)] if len(have) else ext
    new = new.groupby("r1", sort=False).head(topk)   # already sorted by (r1, w desc) inside candidates()
    return new.reset_index(drop=True)


class _KeyView:
    """minimal Side-like view over one key set, so candidates() can run on the keys_v3 arrays unchanged"""
    def __init__(self, key, krow, kw, n):
        self.key, self.krow, self.kw, self._n = key, krow, kw, n

    def __len__(self):
        return self._n
