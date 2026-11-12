"""Shared helpers for the audit: parse a ticket and derive each label field from SCHEMA.md rules.

Everything here uses only SCHEMA.md and the ticket text. No dev/test labels are used.
"""
import json, re, datetime as dt
from decimal import Decimal

ROOT = "E:/Fine_Tuneing-Assigh/training_assessment/candidate_pack/data/"
KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
NULL_CATS = {"account_access", "kyc", "other"}  # transaction fields must be null for these


def load(name):
    import os
    path = name if os.path.isabs(name) else ROOT + name
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- ticket parsing
HDR = re.compile(r"^Received: (\d{4})-(\d{2})-(\d{2}) \((\w{3})\)\nSource: ([\w-]+)\n\n", re.S)


def parse_ticket(user):
    m = HDR.match(user)
    received = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    body = user[m.end():]
    own = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith(">"))  # drop quoted earlier emails
    return received, m.group(5), body, own


# ---------------------------------------------------------------- amounts
NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?"
UNIT = r"k|lakhs?|lacs?"
AMT = re.compile(
    rf"(?P<pre>\u20b9|\bRs\.?|\bINR)\s*(?P<n1>{NUM})(?:\s*(?P<u1>{UNIT})\b)?"
    rf"|(?<![A-Za-z0-9.,])(?P<n2>{NUM})\s*(?P<u2>{UNIT})\b"
    rf"|(?<![A-Za-z0-9.,])(?P<n3>{NUM})\s*(?:rupees|rupaye|rupaiye)\b",
    re.I,
)
DISTRACT_LEFT = re.compile(r"balance|daily limit|limit|\bfees?\b|charges?\b|\bmein abhi\b|\babhi\b", re.I)
DISTRACT_RIGHT = re.compile(r"^\s*(?:right now|pade hain|baki|bacha)", re.I)


def to_paise(num, unit):
    v = Decimal(num.replace(",", ""))
    u = (unit or "").lower()
    if u == "k":
        v *= 1000
    elif u.startswith("lakh") or u.startswith("lac"):
        v *= 100000
    return int(v * 100)


def amounts(text):
    """All currency-like amounts in text -> list of dict(paise, raw, distractor)."""
    out = []
    for m in AMT.finditer(text):
        num = m.group("n1") or m.group("n2") or m.group("n3")
        unit = m.group("u1") or m.group("u2")
        left = text[: m.start()]
        cut = max(left.rfind(". "), left.rfind("! "), left.rfind("? "), left.rfind("\n"))
        sent_left = left[cut + 1:]
        dis = bool(DISTRACT_LEFT.search(sent_left)) or bool(DISTRACT_RIGHT.match(text[m.end():]))
        out.append({"paise": to_paise(num, unit), "raw": m.group(0), "distractor": dis})
    return out


# ---------------------------------------------------------------- dates
MON = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
MONRE = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
ABS_NUM = re.compile(r"(?<![\d/-])(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})(?![\d/-])")
ABS_DM = re.compile(rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+{MONRE}\b(?:,?\s+(\d{{4}}))?", re.I)
ABS_MD = re.compile(
    rf"\b{MONRE}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!\s*:)(?!\.\d)(?!\s*(?:k\b|lakh|lac\b|rupe))(?:,?\s+(\d{{4}}))?", re.I
)  # "Aug 14" / "August 14th 2026"; not "Aug. 1.3k" (a sentence end followed by an amount)
REL = [
    (re.compile(r"day before yesterday|\bparso\b", re.I), -2),
    (re.compile(r"(?<!day before )\byesterday\b|\bkal\b", re.I), -1),
    (re.compile(r"\btoday\b|\baaj\b", re.I), 0),
]
REL_N = re.compile(r"\b(\d+)\s+(?:days?\s+ago|din\s+pehle)", re.I)


def _safe(y, m, d):
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def _no_year(received, m, d):
    c = _safe(received.year, m, d)
    if c is None:
        return None
    return c if c <= received else _safe(received.year - 1, m, d)


def dates(text, received):
    """Candidate transaction dates mentioned in text -> list of (date, kind, raw)."""
    out = []
    for m in ABS_NUM.finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        y = int(y) + 2000 if len(y) == 2 else int(y)
        dd = _safe(y, mo, d)
        if dd:
            out.append((dd, "abs_num", m.group(0)))
    for m in ABS_DM.finditer(text):
        d, mo, y = int(m.group(1)), MON[m.group(2)[:3].lower()], m.group(3)
        dd = _safe(int(y), mo, d) if y else _no_year(received, mo, d)
        if dd:
            out.append((dd, "abs_dm", m.group(0)))
    for m in ABS_MD.finditer(text):
        mo, d, y = MON[m.group(1)[:3].lower()], int(m.group(2)), m.group(3)
        dd = _safe(int(y), mo, d) if y else _no_year(received, mo, d)
        if dd:
            out.append((dd, "abs_md", m.group(0)))
    for rx, off in REL:
        for m in rx.finditer(text):
            out.append((received + dt.timedelta(days=off), "rel", m.group(0)))
    for m in REL_N.finditer(text):
        out.append((received - dt.timedelta(days=int(m.group(1))), "rel", m.group(0)))
    return out


# ---------------------------------------------------------------- txn id
ID_RX = re.compile(r"(?<![A-Za-z])N\s*P(?:[\s-]*\d){10}(?!\d)", re.I)


def txn_ids(text):
    return ["NP" + re.sub(r"\D", "", m.group(0)) for m in ID_RX.finditer(text)]


# ---------------------------------------------------------------- channel
CH_STRICT = [
    (re.compile(r"Mode:\s*(upi|net ?banking|internet banking|wallet|credit card|debit card|card)\b", re.I), 1),
    (re.compile(r"It was an? (upi|net ?banking|internet banking|wallet|credit card|debit card|card) transaction", re.I), 1),
    (re.compile(r"\b(upi|net ?banking|wallet|card|credit card|debit card) se hua tha", re.I), 1),
    (re.compile(r"\b(credit card|debit card|card) wala transaction tha", re.I), 1),
    (re.compile(r"(from the NimbusPay wallet)", re.I), 1),
]
CH_LOOSE = [
    ("upi", re.compile(r"\bUPI\b", re.I)),
    ("netbanking", re.compile(r"net\s?banking|internet banking", re.I)),
    ("wallet", re.compile(r"\bwallet\b", re.I)),
    ("card", re.compile(r"credit card|debit card|card payment|card transaction|\bvisa\b|mastercard|\bcard\b", re.I)),
]


def _canon_ch(s):
    s = s.lower()
    if "upi" in s:
        return "upi"
    if "bank" in s:
        return "netbanking"
    if "wallet" in s:
        return "wallet"
    return "card"


def channels_strict(text):
    return sorted({_canon_ch(m.group(1)) for rx, _ in CH_STRICT for m in rx.finditer(text)})


def channels_loose(text):
    return sorted({n for n, rx in CH_LOOSE if rx.search(text)})


# ---------------------------------------------------------------- language, contact, legal
HI_WORDS = set("""hai hain hoon tha thi the kya kaise kyun kyu nahi nahin mera meri mere mujhe aap apna apni ko ka ki ke se par lekin
paise paisa wapas kat kaat gaye gaya hua hui ho kar karo karna karun kiya raha rahi rahe abhi tak jaldi dhanyavaad namaste bhai ji baar
teesri pehle pichla din rakam kitna kinta chahiye jaunga dunga bol bata batao dikha dikh mein kuch koi aur bhi hi toh to ek do
saamaan ghar phir fir chuka chala chali samajh hua hoga kijiye kijie kare kar sakta sakte""".split())
HI_STRONG = set("""hai hain hoon kya kaise nahi nahin mera meri mere mujhe lekin paise paisa wapas kat kaat gaye gaya raha rahi abhi jaldi
dhanyavaad namaste baar teesri pehle rakam kitna chahiye jaunga dunga bol bata karo karna karun kiya hua tha hoon""".split())


def hindi_score(text):
    toks = re.findall(r"[a-z]+", text.lower())
    return sum(t in HI_STRONG for t in toks), len(toks)


CONTACT = re.compile(
    r"third time|teesri baar|already written|contacted support|second complaint|again and again|baar baar|pehle bhi complaint",
    re.I,
)
TICKET_REF = re.compile(r"earlier ticket number|pichla ticket number", re.I)
LEGAL = re.compile(r"\bRBI\b|ombudsman|consumer court|lawyer|legal notice|legal action", re.I)


# ---------------------------------------------------------------- rules from SCHEMA.md
def exp_priority(cat, amount, legal):
    if cat == "fraud" or (amount is not None and amount >= 5_000_000) or legal:
        return "P1"
    if (cat in ("payment_failed", "refund") and amount is not None and amount >= 500_000) or cat == "account_access":
        return "P2"
    return "P3"


def exp_needs_human(priority, contacted):
    return priority == "P1" or contacted


def parse_label(content):
    try:
        lab = json.loads(content)
    except Exception:
        return None
    return lab if isinstance(lab, dict) else None


# ---------------------------------------------------------------- full derivation (spec + text -> expected label fields)
SCRATCH = re.compile(r"scratch\s*card|srcatch\s*card", re.I)


def channel_from_text(own):
    st = channels_strict(own)
    if len(st) == 1:
        return st[0]
    if len(st) > 1:
        return None
    lo = channels_loose(SCRATCH.sub(" ", own))
    return lo[0] if len(lo) == 1 else None


def derive(user, category):
    """Expected values for every deterministic field, given the (trusted) category.

    Returns dict with the fields plus 'ambiguous' = list of fields where the text gave >1 candidate.
    """
    rec, src, body, own = parse_ticket(user)
    amb = []
    if category in NULL_CATS:
        amount = date = tid = channel = None
    else:
        main = sorted({a["paise"] for a in amounts(own) if not a["distractor"]})
        if len(main) > 1:
            amb.append("amount_paise")
        amount = main[0] if len(main) == 1 else None
        cands = sorted({d.isoformat() for d, _, _ in dates(own, rec)})
        if len(cands) > 1:
            amb.append("txn_date")
        date = cands[0] if len(cands) == 1 else None
        ids = sorted(set(txn_ids(own)))
        if len(ids) > 1:
            amb.append("txn_id")
        tid = ids[0] if len(ids) == 1 else None
        channel = channel_from_text(own)
    legal, contacted = bool(LEGAL.search(own)), bool(CONTACT.search(own))
    prio = exp_priority(category, amount, legal)
    hs, _ = hindi_score(own)
    return {
        "category": category, "priority": prio, "amount_paise": amount, "txn_date": date, "txn_id": tid,
        "channel": channel, "language": "hi-en" if hs >= 2 else "en",
        "needs_human": exp_needs_human(prio, contacted), "ambiguous": amb,
        "_legal": legal, "_contacted": contacted,
    }
