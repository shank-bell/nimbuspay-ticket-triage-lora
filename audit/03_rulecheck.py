"""Part A, step 3: check every training label against rules derived from SCHEMA.md + the ticket text.

Usage: python 03_rulecheck.py [dataset]   (default train.jsonl; can also be dev.jsonl to validate the checker itself)
Writes audit/violations_<dataset>.csv  (id, field, label, expected, note)
"""
import csv, sys, os
from collections import Counter, defaultdict
from nimbus_lib import *

ds = sys.argv[1] if len(sys.argv) > 1 else "train.jsonl"
rows = load(ds)
viol = []  # (id, field, got, expected, note)
n_checked = 0


def add(rid, field, got, exp, note=""):
    viol.append((rid, field, got, exp, note))


for r in rows:
    rid = r["id"]
    lab = parse_label(r["messages"][-1]["content"])
    if lab is None:
        add(rid, "json", "unparseable", "valid json", "truncated")
        continue
    if set(lab) != set(KEYS):
        add(rid, "keys", sorted(lab), KEYS, "schema keys differ (legacy?)")
        continue
    n_checked += 1
    received, source, body, own = parse_ticket(r["messages"][1]["content"])
    cat = lab["category"]

    # enum / type checks
    if cat not in ("payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"):
        add(rid, "category", cat, "enum", "bad enum")
    if lab["priority"] not in ("P1", "P2", "P3"):
        add(rid, "priority", lab["priority"], "enum", "bad enum")
    if lab["language"] not in ("en", "hi-en"):
        add(rid, "language", lab["language"], "enum", "bad enum")
    if lab["channel"] not in (None, "upi", "card", "netbanking", "wallet"):
        add(rid, "channel", lab["channel"], "enum", "bad enum")
    if lab["amount_paise"] is not None and (type(lab["amount_paise"]) is not int):
        add(rid, "amount_paise", lab["amount_paise"], "int", "bad type")
    if lab["txn_date"] is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(lab["txn_date"])):
        add(rid, "txn_date", lab["txn_date"], "YYYY-MM-DD", "bad format")
    if lab["txn_id"] is not None and not re.fullmatch(r"NP\d{10}", str(lab["txn_id"])):
        add(rid, "txn_id", lab["txn_id"], "NP+10 digits", "bad format")

    # null rule
    if cat in NULL_CATS:
        for k in ("amount_paise", "txn_date", "txn_id", "channel"):
            if lab[k] is not None:
                add(rid, k, lab[k], None, f"must be null for {cat}")

    # ---- priority / needs_human consistency (computed from the label's own category + amount + text)
    legal = bool(LEGAL.search(own))
    contacted = bool(CONTACT.search(own))
    ep = exp_priority(cat, lab["amount_paise"], legal)
    if lab["priority"] != ep:
        add(rid, "priority", lab["priority"], ep, f"cat={cat} amt={lab['amount_paise']} legal={legal}")
    enh = exp_needs_human(lab["priority"], contacted)
    if lab["needs_human"] != enh:
        add(rid, "needs_human", lab["needs_human"], enh, f"prio={lab['priority']} contacted={contacted}")

    # ---- amount
    if cat not in NULL_CATS:
        am = amounts(own)
        main = [a["paise"] for a in am if not a["distractor"]]
        allv = [a["paise"] for a in am]
        if lab["amount_paise"] is None:
            if main:
                add(rid, "amount_paise", None, main, "text has an amount")
        else:
            if lab["amount_paise"] not in allv:
                add(rid, "amount_paise", lab["amount_paise"], main or allv, "label amount not found in text")
            elif lab["amount_paise"] not in main:
                add(rid, "amount_paise", lab["amount_paise"], main, "label uses a distractor amount")
            elif len(set(main)) > 1:
                add(rid, "amount_paise", lab["amount_paise"], main, "ambiguous: several main amounts")

    # ---- date
    if cat not in NULL_CATS:
        ds_ = dates(own, received)
        cands = sorted({d.isoformat() for d, _, _ in ds_})
        if lab["txn_date"] is None:
            if cands:
                add(rid, "txn_date", None, cands, "text has a date: " + "; ".join(f"{k}:{raw}" for _, k, raw in ds_))
        else:
            if lab["txn_date"] not in cands:
                add(rid, "txn_date", lab["txn_date"], cands, "; ".join(f"{k}:{raw}" for _, k, raw in ds_) or "no date in text")
            elif len(cands) > 1:
                add(rid, "txn_date", lab["txn_date"], cands, "ambiguous: several candidate dates")

    # ---- txn id
    if cat not in NULL_CATS:
        ids = sorted(set(txn_ids(own)))
        if lab["txn_id"] is None:
            if ids:
                add(rid, "txn_id", None, ids, "text has an NP id")
        elif lab["txn_id"] not in ids:
            add(rid, "txn_id", lab["txn_id"], ids, "label id not in text (normalised)")

    # ---- channel (explicit mode phrase, or an explicit "credit card / card payment / my card was declined" mention)
    if cat not in NULL_CATS:
        exp_ch = channel_from_text(own)
        if lab["channel"] != exp_ch:
            add(rid, "channel", lab["channel"], exp_ch, f"strict={channels_strict(own)} loose={channels_loose(own)}")

    # ---- language
    hs, nt = hindi_score(own)
    hi = hs >= 2
    if (lab["language"] == "hi-en") != hi:
        if not (hs == 1):  # borderline cases are reported separately
            add(rid, "language", lab["language"], "hi-en" if hi else "en", f"hindi_score={hs}/{nt}")
        else:
            add(rid, "language", lab["language"], "?", f"borderline hindi_score=1/{nt}")

by_field = Counter(v[1] for v in viol)
print(f"{ds}: rows={len(rows)} checked_full_schema={n_checked}  total_flags={len(viol)}")
for f, c in by_field.most_common():
    print(f"  {f:14} {c}")
with open(f"E:/Fine_Tuneing-Assigh/audit/violations_{os.path.basename(ds).split('.')[0]}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "field", "label", "expected", "note"])
    w.writerows(viol)
