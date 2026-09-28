"""Text normalisation for business names and addresses (no external data)."""
from functools import lru_cache
import re, unicodedata

LEGAL = {"llc","inc","ltd","limited","private","pvt","corp","corporation","co","company",
         "llp","pc","plc","lp","the","and","of","pllc","incorporated","opc","ll","p","l",
         # French legal-entity suffixes (test set adds France, unseen at training time;
         # without these "sarl"/"sas"/etc. were treated as real name content, corrupting
         # both blocking keys and every name-similarity feature for every France pair)
         "sarl","sas","sasu","eurl","sa","sci","snc","scop","scea","sca","gie","eirl",
         "selarl","selas","selasu","groupement",
         # a handful of other common international suffixes, defensively, since the
         # problem statement says country is an open set and must not be hard-coded
         "gmbh","ag","kg","ohg","mbh","srl","sl","spa","bv","nv","oy","ab","as","kft"}
ABBR = {"dr":"drive","rd":"road","st":"street","ave":"avenue","av":"avenue","blvd":"boulevard",
        "ln":"lane","ct":"court","hwy":"highway","pkwy":"parkway","cir":"circle","pl":"place",
        "ter":"terrace","trl":"trail","sq":"square","mt":"mount","ft":"fort","n":"north",
        "s":"south","e":"east","w":"west","nr":"near","opp":"opposite","cross":"cross",
        # French street-type abbreviations, so "R." / "Bd" / "Che" line up with the
        # unabbreviated "rue"/"boulevard"/"chemin" spelled out on the other source
        "r":"rue","bd":"boulevard","che":"chemin","chem":"chemin","all":"allee",
        "imp":"impasse","fg":"faubourg","pas":"passage","res":"residence"}
ADDR_STOP = {"unit","apartment","apt","floor","fl","suite","ste","near","opposite","no","nd","th",
             "rd","st","po","box","block","building","bldg","room","flat","plot","house","shop",
             "hno","sno","the","and"}
US_STATES = {"alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca","colorado":"co",
 "connecticut":"ct","delaware":"de","florida":"fl","georgia":"ga","hawaii":"hi","idaho":"id","illinois":"il",
 "indiana":"in","iowa":"ia","kansas":"ks","kentucky":"ky","louisiana":"la","maine":"me","maryland":"md",
 "massachusetts":"ma","michigan":"mi","minnesota":"mn","mississippi":"ms","missouri":"mo","montana":"mt",
 "nebraska":"ne","nevada":"nv","new hampshire":"nh","new jersey":"nj","new mexico":"nm","new york":"ny",
 "north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok","oregon":"or","pennsylvania":"pa",
 "rhode island":"ri","south carolina":"sc","south dakota":"sd","tennessee":"tn","texas":"tx","utah":"ut",
 "vermont":"vt","virginia":"va","washington":"wa","west virginia":"wv","wisconsin":"wi","wyoming":"wy"}
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(US_STATES, key=len, reverse=True)) + r")\b")
_NONWORD = re.compile(r"[^\w\s]", re.U)
_SPACE = re.compile(r"\s+")


_INDIC = ("DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM")
_VOW = {"A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "E": "e", "EE": "e", "SHORT E": "e",
        "CANDRA E": "e", "AI": "ai", "O": "o", "OO": "o", "SHORT O": "o", "CANDRA O": "o", "AU": "au",
        "VOCALIC R": "ri", "VOCALIC RR": "ri", "VOCALIC L": "li", "SHORT A": "a", "CANDRA A": "a", "AW": "au",
        "OE": "o", "UE": "u", "PRISHTHAMATRA E": "e", "AI LENGTH MARK": "", "AU LENGTH MARK": ""}
_CONS = {"CA": "ch", "CHA": "chh", "SSA": "sh", "SHA": "sh", "TTA": "t", "TTHA": "th", "DDA": "d", "DDHA": "dh",
         "NNA": "n", "NNNA": "n", "LLA": "l", "LLLA": "l", "RRA": "r", "ZHA": "l", "KSSA": "ksh", "JNYA": "gy",
         "NYA": "ny", "NGA": "ng", "VA": "v", "FA": "f", "ZA": "z", "NUKTA": ""}


def translit(s: str) -> str:
    """Rough Indic-script -> Latin using only Unicode character names (no external data)."""
    out, last_cons = [], False
    for ch in s:
        if ch < "ऀ":
            out.append(ch); last_cons = False; continue
        try:
            nm = unicodedata.name(ch)
        except ValueError:
            out.append(" "); last_cons = False; continue
        sc = nm.split(" ", 1)[0]
        if sc not in _INDIC:
            out.append(ch); last_cons = False; continue
        rest = nm[len(sc) + 1:]
        if rest.startswith("LETTER "):
            L = rest[7:]
            if L in _VOW:
                out.append(_VOW[L]); last_cons = False
            else:
                b = _CONS.get(L) if L in _CONS else (L[:-1] if L.endswith("A") and len(L) > 1 else L)
                out.append(b.lower() + "a"); last_cons = True
        elif rest.startswith("VOWEL SIGN "):
            v = _VOW.get(rest[11:], "a")
            if last_cons and out and out[-1].endswith("a"): out[-1] = out[-1][:-1]
            out.append(v); last_cons = False
        elif rest == "SIGN VIRAMA":
            if last_cons and out and out[-1].endswith("a"): out[-1] = out[-1][:-1]
            last_cons = False
        elif rest in ("SIGN ANUSVARA", "SIGN CANDRABINDU", "SIGN BINDI", "SIGN TIPPI", "SIGN ADDAK"):
            out.append("n"); last_cons = False
        elif rest == "SIGN VISARGA":
            out.append("h"); last_cons = False
        elif "DIGIT" in rest:
            out.append(str(unicodedata.digit(ch, ""))); last_cons = False
        else:
            out.append(" " if "DANDA" in rest else ""); last_cons = False
    return "".join(out)


LEGAL_SK = {"prvt", "lmtd", "lmt", "pr", "l", "prl", "ltd", "llp", "kmpn", "kp", "prvtl", "lmtdd"}


def learn_legal_sk(names, min_end_frac=0.005, max_len=10):
    """Data-driven legal suffix learner (feat_v3 only, never called when feat_v3 is off).

    Scans all S1 business_name values for the country and finds token skeletons that
    appear at the END of a name more than `min_end_frac` of the time.  Learned entirely
    from the data — no language-specific lists, so it works for any unseen country.

    Algorithm:
      1. Normalise every name through norm_name().
      2. Count each token skeleton that appears as the LAST token of any name.
      3. Keep skeletons whose end-of-name frequency >= min_end_frac * total_names AND
         whose token length <= max_len (avoids picking up real short words like "the").
      4. Union with the hard-coded LEGAL_SK so existing coverage is never lost.

    Returns a frozenset of skeleton strings (superset of LEGAL_SK).
    Prints the newly learned tokens so they can be audited.
    """
    from collections import Counter
    total = 0
    end_cnt = Counter()
    for nm in names:
        _, core = norm_name(nm)
        if not core:
            continue
        total += 1
        sk = skel(core[-1])
        if sk and len(sk) <= max_len:
            end_cnt[sk] += 1
    threshold = max(2, int(min_end_frac * total))
    new_sk = frozenset(s for s, c in end_cnt.items() if c >= threshold) - LEGAL_SK
    if new_sk:
        print(f"  [feat_v3] learn_legal_sk: {total:,} names, threshold={threshold} "
              f"-> {len(new_sk)} new suffix skeletons: {sorted(new_sk)}")
    else:
        print(f"  [feat_v3] learn_legal_sk: {total:,} names, threshold={threshold} -> no new skeletons beyond LEGAL_SK")
    return LEGAL_SK | new_sk
_SKR = [("ph", "f"), ("kh", "k"), ("gh", "g"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("jh", "j"), ("sh", "s"),
        ("ch", "k"), ("ck", "k"), ("qu", "k"), ("x", "ks"), ("c", "k"), ("q", "k"), ("w", "v"), ("z", "j"), ("g", "j")]


@lru_cache(maxsize=1 << 18)
def skel(tok: str) -> str:
    """consonant skeleton: crude phonetic key comparable between Latin and transliterated Indic text"""
    for a, b in _SKR:
        tok = tok.replace(a, b)
    tok = re.sub(r"[aeiouyh]", "", tok)
    return re.sub(r"(.)\1+", r"\1", tok)


def strip_accents(s: str) -> str:
    """Remove accents / diacritics (NFKD + drop combining marks); ASCII text is returned as is."""
    if s.isascii():  # NFKD never changes ASCII and ASCII has no combining marks -> same result
        return s
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_name(s: str, extra_legal_sk: frozenset = frozenset()):
    """returns (clean string, core tokens list without legal suffixes).
    extra_legal_sk: additional skeleton strings to treat as legal suffixes (feat_v3 only).
    When empty (the default, used by all existing call sites), behaviour is identical to before."""
    if s and max(s) >= "ऀ":  # same as any(c >= "ऀ" for c in s), done in C
        s = translit(s)
    s = strip_accents(s).lower().replace("&", " and ").replace("'", "").replace(".", "")
    s = _SPACE.sub(" ", _NONWORD.sub(" ", s)).strip()
    toks = s.split()
    _lsk = LEGAL_SK | extra_legal_sk
    core = [t for t in toks if t not in LEGAL and skel(t) not in _lsk] or toks
    return " ".join(toks), core


ORD = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
       "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th", "eleventh": "11th",
       "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
       "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th", "nineteenth": "19th",
       "twentieth": "20th", "thirtieth": "30th", "fortieth": "40th", "fiftieth": "50th"}


def norm_addr(s: str, ords=False):
    """returns (clean string, tokens list). Expands abbreviations, strips leading zeros.
    ords=True (keys_v2 configs only): spelled-out ordinals -> digits ('thirteenth' -> '13th')."""
    if s and max(s) >= "ऀ":  # same as any(c >= "ऀ" for c in s), done in C
        s = translit(s)
    s = strip_accents(s).lower().replace(".", "")
    s = _STATE_RE.sub(lambda m: US_STATES[m.group(1)], s)
    s = _SPACE.sub(" ", _NONWORD.sub(" ", s)).strip()
    out = []
    for t in s.split():
        t = ABBR.get(t, t)
        if ords:
            t = ORD.get(t, t)
        if t.isdigit():
            t = t.lstrip("0") or "0"
        elif t[0].isdigit():
            t = t.lstrip("0")
        out.append(t)
    return " ".join(out), out


@lru_cache(maxsize=1 << 18)
def _hasdig(t):
    """True if the token contains a digit (cached: address tokens repeat a lot)."""
    return any(c.isdigit() for c in t)


def addr_keys(toks, max_keys=3, stop=frozenset()):
    """(number, following street word) keys, e.g. ('41','groton').
    `stop`: extra generic words to skip (see generic_addr_tokens). Without it, a French address
    '34 Rue Frederic Bastiat' gives the useless key ('34','rue') shared by every '34 Rue ...'."""
    keys = []
    look = 6 if stop else 4  # skipping generic words needs a slightly longer lookahead
    for i, t in enumerate(toks):
        if _hasdig(t) and len(t) <= 8:
            for j in range(i + 1, min(i + look, len(toks))):
                w = toks[j]
                if w.isalpha() and len(w) > 2 and w not in ADDR_STOP and w not in stop:
                    keys.append(t + "|" + w); break
            if len(keys) >= max_keys:
                break
    return keys


def generic_addr_tokens(addresses, frac=0.01, max_n=200_000, seed=0):
    """Data-driven generic address words for ONE country: alphabetic tokens present in more than
    `frac` of that country's addresses (street types, articles, city/state names: 'street',
    'sector', 'rue', 'de', ...). Learned from the data itself, so it works for any country label,
    including ones never seen in training (France), without hard-coding any language."""
    import random
    addrs = list(addresses)
    if len(addrs) > max_n:
        addrs = random.Random(seed).sample(addrs, max_n)
    cnt = {}
    for a in addrs:
        for t in set(norm_addr(a)[1]):
            if t.isalpha():
                cnt[t] = cnt.get(t, 0) + 1
    n = max(len(addrs), 1)
    return frozenset(t for t, c in cnt.items() if c / n > frac)
