"""Build a training sample that keeps the *density of look-alike businesses* of the real data.

S1 entities are chosen by a hash of their normalised name (so all businesses sharing a name are kept together),
together with their true S2/S3 matches and the unmatched S2/S3 rows of the same name buckets.

    python experiments/make_sample.py --data ../../dataset/train --out ../../sample_dense --mod 20
"""
import os as _os, sys as _sys  # experiments/ scripts import the pipeline modules from ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import argparse, os, zlib
from norm import norm_name


def bucket(name, country, mod):
    core = " ".join(sorted(norm_name(name)[1]))
    return zlib.crc32((country + "|" + core).encode("utf-8")) % mod


def rows(path):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        head = f.readline().rstrip("\r\n")
        yield head
        for ln in f:
            yield ln.rstrip("\r\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mod", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    keep_s1, s1_lines, head1 = set(), [], None
    it = rows(os.path.join(a.data, "train_source1.tsv")); head1 = next(it)
    for ln in it:
        p = ln.split("\t")
        if len(p) >= 4 and bucket(p[1], p[3], a.mod) == 0:
            keep_s1.add(p[0]); s1_lines.append(ln)
    print("S1 kept:", len(keep_s1))
    it = rows(os.path.join(a.data, "train_ground_truth.tsv")); headg = next(it)
    gt_lines, keep_o = [], set()
    for ln in it:
        p = ln.split("\t")
        if p[0] in keep_s1:
            gt_lines.append(ln)
            if len(p) > 1 and p[1]:
                keep_o.update(p[1].split(","))
    for name in ("train_source2.tsv", "train_source3.tsv"):
        it = rows(os.path.join(a.data, name)); h = next(it); n = 0
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="\n") as out:
            out.write(h + "\n")
            for ln in it:
                p = ln.split("\t")
                if p[0] in keep_o or (len(p) >= 4 and bucket(p[1], p[3], a.mod) == 0):
                    out.write(ln + "\n"); n += 1
        print(name, n)
    for name, head, lines in (("train_source1.tsv", head1, s1_lines), ("train_ground_truth.tsv", headg, gt_lines)):
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="\n") as out:
            out.write(head + "\n"); out.write("\n".join(lines) + "\n")
    print("done ->", a.out)


if __name__ == "__main__":
    main()
