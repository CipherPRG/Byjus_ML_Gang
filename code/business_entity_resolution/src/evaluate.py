"""The competition metric: macro-averaged F0.5 over Source-1 entities."""
def f05_macro(pred, truth):
    """Macro F0.5 over S1 entities. pred/truth: dict s1_id -> set(ids). empty==empty counts 1.0."""
    tot = 0.0
    for k, t in truth.items():
        p = pred.get(k, set())
        if not t and not p:
            tot += 1.0
            continue
        tp = len(p & t)
        if tp == 0:
            continue
        pr, rc = tp / len(p), tp / len(t)
        tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / max(len(truth), 1)
