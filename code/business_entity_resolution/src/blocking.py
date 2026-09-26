"""Blocking / candidate generation. Country-scoped keys, numpy/pandas merges (memory-lean)."""
import zlib
import numpy as np
import pandas as pd
from norm import norm_name, norm_addr, addr_keys, ADDR_STOP, skel

# weight of each key type (address keys are the most reliable)
WEIGHT = {"A": 3, "R": 2, "N": 3, "Q": 2, "P": 2, "T": 1, "U": 1, "S": 1, "K": 1, "V": 2, "X": 3}


def keys_for(country, core, atoks):
    ks = []
    for k in addr_keys(atoks):
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
    ak += ["w" + t for t in sorted({t for t in atoks if len(t) >= 5 and t.isalpha() and t not in ADDR_STOP},
                                   key=lambda t: (-len(t), t))[:2]]
    for a_ in nk[:4]:
        for b_ in ak[:4]:
            ks.append("X|" + country + "|" + a_ + "|" + b_)
    al = sorted({t for t in atoks if len(t) >= 5 and t.isalpha() and t not in ADDR_STOP},
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

    def __init__(self, df):
        self.ids = df.entity_id.tolist()
        self.nclean, self.aclean, self.nskel = [], [], []
        key, row, w = [], [], []
        for i, (nm, ad, c) in enumerate(zip(df.business_name.values, df.business_address.values, df.country.values)):
            nc, core = norm_name(nm)
            ac, at = norm_addr(ad)
            self.nclean.append(nc)
            self.nskel.append(" ".join(x for x in (skel(t) for t in core) if x))
            self.aclean.append(ac)
            for k in keys_for(c, core, at):
                key.append(h64(k)); row.append(i); w.append(WEIGHT[k[0]])
        self.key = np.array(key, dtype=np.int64)
        self.krow = np.array(row, dtype=np.int32)
        self.kw = np.array(w, dtype=np.int8)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def concat(sides):
        out = Side.__new__(Side)
        out.ids, out.nclean, out.aclean, out.nskel = [], [], [], []
        keys, rows, ws = [], [], []
        off = 0
        for s in sides:
            out.ids += s.ids; out.nclean += s.nclean; out.aclean += s.aclean; out.nskel += s.nskel
            keys.append(s.key); rows.append(s.krow + off); ws.append(s.kw); off += len(s)
        out.key = np.concatenate(keys); out.krow = np.concatenate(rows); out.kw = np.concatenate(ws)
        return out


def candidates(s1, oth, max_block=60, max_s1_block=200, topk=60, batch=50_000):
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
