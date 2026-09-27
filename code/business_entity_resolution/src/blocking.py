"""Blocking / candidate generation. Country-scoped keys, numpy/pandas merges (memory-lean)."""
import zlib
import numpy as np
import pandas as pd
from norm import norm_name, norm_addr, addr_keys, ADDR_STOP, skel

# weight of each key type (address keys are the most reliable)
WEIGHT = {"A": 3, "R": 2, "N": 3, "Q": 2, "P": 2, "T": 1, "U": 1, "S": 1, "K": 1, "V": 2, "X": 3,
          "B": 2, "Y": 2, "C": 3, "W": 2}
_DOM = ("com", "net", "org")
_NAME_STOP_CACHE = {}


def compute_name_stop(names, frac=0.002):
    """Compute per-country generic name core tokens from an iterable of business names:
    alphabetic tokens present in > frac of the names (industry words: 'services', 'systems', ...)."""
    cnt = {}
    for nm in names:
        _, core = norm_name(nm)
        for t in set(core):
            if len(t) >= 4 and t.isalpha():
                cnt[t] = cnt.get(t, 0) + 1
    n = max(len(names), 1)
    return frozenset(t for t, c in cnt.items() if c / n > frac)


def init_name_stop(country, names, frac=0.002):
    """Compute and cache dynamic per-country generic name core tokens up front
    (e.g. from S1+S2+S3 combined names) before any Side is constructed."""
    if isinstance(names, (list, tuple)) and names and hasattr(names[0], "__iter__") and not isinstance(names[0], (str, bytes)):
        import itertools
        names = list(itertools.chain.from_iterable(names))
    _NAME_STOP_CACHE[country] = compute_name_stop(names, frac=frac)
    return _NAME_STOP_CACHE[country]


def get_name_stop(country, names=None, frac=0.002):
    """Retrieve cached name_stop for country. If not cached:
    - If names is provided, computes, caches, and returns it.
    - If names is None, raises RuntimeError to prevent order-dependent empty set."""
    if country not in _NAME_STOP_CACHE:
        if names is None:
            raise RuntimeError(
                f"name_stop for country '{country}' has not been initialized. "
                f"Call init_name_stop('{country}', names) before constructing Side, "
                f"or pass names directly."
            )
        return init_name_stop(country, names, frac=frac)
    return _NAME_STOP_CACHE[country]


def _hasdig(t):
    return any(ch.isdigit() for ch in t)


def keys_v2(country, core, atoks, stop, is_other=False, name_stop=frozenset()):
    """Extra keys (cfg keys_v2=True), each aimed at a blocking-miss pattern seen on sample data:
    B  adjacent address bigram containing a number: '4600 24th', '5534 10480', 'c 25', '14 109'
    Y  unordered (number, rare address word): 'no226|villupuram' even when word order differs
    C  compact name: 'allshivsystems' / initials+last 'wmbrokerage', matching website-style names
       'allshivsystemscom' / 'wmbrokeragecom' (single token ending in com/net/org, suffix stripped)
    W  empty-address resilient core-token key: gated to fire only when other-side address is empty/<=1 token"""
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
    # W key: empty-address resilient core-token key
    distinct_core = [t for t in core if len(t) >= 5 and t.isalpha() and t not in name_stop]
    top_core = sorted(set(distinct_core), key=lambda x: (-len(x), x))[:2]
    if is_other:
        if len(atoks) <= 1:
            for t in top_core:
                ks.append("W|" + country + "|" + t)
    else:
        for t in top_core:
            ks.append("W|" + country + "|" + t)
    return ks


def keys_for(country, core, atoks, stop=frozenset(), kv2=False, is_other=False, name_stop=frozenset()):
    ks = keys_v2(country, core, atoks, stop, is_other=is_other, name_stop=name_stop) if kv2 else []
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
    ak = ["d" + t for t in [t for t in atoks if any(ch.isdigit() for ch in t) and len(t) <= 8][:2]]
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
    b = k.encode("utf-8")
    return ((zlib.crc32(b) & 0x7FFFFFFF) << 32) | zlib.adler32(b)


class Side:
    """Normalised records of one source (one country): ids, clean name/address, blocking keys."""

    def __init__(self, df, stop=frozenset(), kv2=False, name_stop=None):
        self.stop = stop  # per-country generic address words (norm.generic_addr_tokens); empty = off
        self.kv2 = kv2    # keys_v2 blocking keys + ordinal normalisation (cfg keys_v2); False = v7 behaviour
        self.ids = df.entity_id.tolist()
        self.nclean, self.aclean, self.nskel = [], [], []
        key, row, w = [], [], []
        first_id = str(df.entity_id.iloc[0]) if len(df) else ""
        is_other = not first_id.startswith("S1")
        c_val = df.country.iloc[0] if len(df) else ""
        if name_stop is not None:
            self.name_stop = name_stop
        elif kv2:
            self.name_stop = get_name_stop(c_val, df.business_name.values if not is_other else None)
        else:
            self.name_stop = frozenset()
        name_stop = self.name_stop
        for i, (nm, ad, c) in enumerate(zip(df.business_name.values, df.business_address.values, df.country.values)):
            nc, core = norm_name(nm)
            ac, at = norm_addr(ad, ords=kv2)
            self.nclean.append(nc)
            self.nskel.append(" ".join(x for x in (skel(t) for t in core) if x))
            self.aclean.append(ac)
            for k in keys_for(c, core, at, stop, kv2, is_other=is_other, name_stop=name_stop):
                key.append(h64(k)); row.append(i); w.append(WEIGHT[k[0]])
        self.key = np.array(key, dtype=np.int64)
        self.krow = np.array(row, dtype=np.int32)
        self.kw = np.array(w, dtype=np.int8)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def concat(sides):
        out = Side.__new__(Side)
        out.stop = sides[0].stop if sides else frozenset()
        out.kv2 = sides[0].kv2 if sides else False
        out.name_stop = sides[0].name_stop if sides else frozenset()
        out.ids, out.nclean, out.aclean, out.nskel = [], [], [], []
        keys, rows, ws = [], [], []
        off = 0
        for s in sides:
            out.ids += s.ids; out.nclean += s.nclean; out.aclean += s.aclean; out.nskel += s.nskel
            keys.append(s.key); rows.append(s.krow + off); ws.append(s.kw); off += len(s)
        out.key = np.concatenate(keys); out.krow = np.concatenate(rows); out.kw = np.concatenate(ws)
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
