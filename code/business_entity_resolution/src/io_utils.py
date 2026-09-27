"""TSV input/output helpers (all columns read as strings, IDs written exactly as given)."""
import pandas as pd

TSV = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3, encoding="utf-8", encoding_errors="replace")


def read_tsv(path, **kw):
    """Read a challenge TSV with every column as a string (no NaN conversion, no quoting)."""
    return pd.read_csv(path, **{**TSV, **kw})


def countries_of(path):
    """Sorted distinct country labels of a TSV (country is an open set)."""
    return sorted(read_tsv(path, usecols=["country"]).country.unique())


def read_country(path, country, chunksize=400_000):
    """Stream a big TSV and keep only rows of one country."""
    parts = [c[c.country == country] for c in read_tsv(path, chunksize=chunksize)]
    parts = [p for p in parts if len(p)]
    if not parts:
        return read_tsv(path, nrows=0)
    return pd.concat(parts, ignore_index=True)


def write_lists(path, header, ids, lists):
    """ids: ordered S1 ids; lists: dict id -> list[str]. Empty list -> empty cell."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header[0] + "\t" + header[1] + "\n")
        for i in ids:
            f.write(i + "\t" + ",".join(lists.get(i, [])) + "\n")
