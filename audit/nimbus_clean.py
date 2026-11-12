"""Cleaning of train.jsonl (Part A). Uses only train.jsonl + SCHEMA.md rules (nimbus_lib.derive).

One exception, disclosed in the report: dev *inputs* (never dev labels) are used to find train rows that are copies of
dev tickets, because training on them would break the "never train on dev" rule.
"""
import json, re, collections
from nimbus_lib import *

JUNK = re.compile(r"^\s*(?:test|asdf asdf|test test \d+)?\s*$", re.I)
LEGACY_PRIO = {"low": "P3", "medium": "P2", "high": "P1"}


def norm_id(s):
    return "NP" + re.sub(r"\D", "", s) if s else s


def clean_row(r):
    """Return (new_label_dict, [fix reasons]) or (None, [drop reasons])."""
    user = r["messages"][1]["content"]
    lab = parse_label(r["messages"][2]["content"])
    fixes = []

    # ---- v1 (legacy) format: prio low/medium/high + dd/mm/yyyy dates
    if "prio" in lab:
        old = lab["prio"]
        lab = {("priority" if k == "prio" else k): v for k, v in lab.items()}
        lab["priority"] = LEGACY_PRIO[old]
        if lab["txn_date"]:
            dd, mm, yy = lab["txn_date"].split("/")
            lab["txn_date"] = f"{yy}-{mm}-{dd}"
        fixes.append(f"legacy v1 format: 'prio':'{old}'->'priority':'{lab['priority']}', date dd/mm/yyyy->ISO")
    lab = {k: lab[k] for k in KEYS}   # canonical key order
    d = derive(user, lab["category"])
    if d["ambiguous"]:
        return None, [f"ambiguous text for {d['ambiguous']}; label cannot be verified"]

    # ---- amount: rupees stored instead of paise
    if lab["amount_paise"] != d["amount_paise"]:
        t = d["amount_paise"]
        if lab["amount_paise"] is not None and t is not None and lab["amount_paise"] == t // 100:
            fixes.append(f"amount_paise stored in rupees ({lab['amount_paise']}) instead of paise ({t})")
            lab["amount_paise"] = t
        else:
            return None, [f"amount_paise {lab['amount_paise']} unexplained (text gives {t})"]
    # ---- txn_id: raw text copied instead of normalised NP+10 digits
    if lab["txn_id"] != d["txn_id"]:
        if lab["txn_id"] is not None and norm_id(lab["txn_id"]) == d["txn_id"]:
            fixes.append(f"txn_id not normalised ('{lab['txn_id']}' -> '{d['txn_id']}')")
            lab["txn_id"] = d["txn_id"]
        else:
            return None, [f"txn_id {lab['txn_id']} unexplained (text gives {d['txn_id']})"]
    # ---- remaining deterministic fields must already agree
    for k in ("txn_date", "channel", "language"):
        if lab[k] != d[k]:
            return None, [f"{k} {lab[k]!r} unexplained (text/spec gives {d[k]!r})"]
    # ---- priority: recompute from spec. Evidence that the *priority* column is the corrupted one:
    #      needs_human always agrees with the spec-derived priority, never with the bad label.
    if lab["needs_human"] != d["needs_human"]:
        return None, [f"needs_human {lab['needs_human']} unexplained (spec gives {d['needs_human']})"]
    if lab["priority"] != d["priority"]:
        fixes.append(
            f"priority {lab['priority']} contradicts spec -> {d['priority']} "
            f"(category={lab['category']}, amount_paise={lab['amount_paise']}, legal_threat={d['_legal']}); needs_human agrees with {d['priority']}"
        )
        lab["priority"] = d["priority"]
    return lab, fixes


def clean_dataset(rows, dev_rows):
    """-> (kept_rows, log[(id, action, reason)], stats Counter)"""
    dev_inputs = {r["messages"][1]["content"]: r["id"] for r in dev_rows}
    log, kept, seen_full = [], [], {}
    stats = collections.Counter()
    for r in rows:
        rid, user = r["id"], r["messages"][1]["content"]
        own = parse_ticket(user)[3] if HDR.match(user) else ""
        if user in dev_inputs:                                   # 1. leakage of dev tickets into train
            log.append((rid, "drop", f"leakage: ticket text identical to dev ticket {dev_inputs[user]} (never train on dev)"))
            stats["drop_dev_leak"] += 1
            continue
        if JUNK.match(own):                                      # 2. junk / empty ticket with an unsupported label
            log.append((rid, "drop", f"junk ticket (body={own.strip()[:15]!r}): label has fields the text cannot support"))
            stats["drop_junk"] += 1
            continue
        if parse_label(r["messages"][2]["content"]) is None:     # 3. truncated label
            log.append((rid, "drop", "assistant label is truncated / invalid JSON"))
            stats["drop_truncated"] += 1
            continue
        if user in seen_full:                                    # 4. exact duplicate of an earlier kept ticket
            log.append((rid, "drop", f"exact duplicate of {seen_full[user]} (same text and label)"))
            stats["drop_duplicate"] += 1
            continue
        lab, notes = clean_row(r)
        if lab is None:
            log.append((rid, "drop", "; ".join(notes)))
            stats["drop_unexplained"] += 1
            continue
        if notes:
            log.append((rid, "fix", "; ".join(notes)))
            for key, pat in (("fix_legacy", "legacy v1"), ("fix_amount", "stored in rupees"), ("fix_txn_id", "not normalised"), ("fix_priority", "contradicts spec")):
                if pat in notes[0]:
                    stats[key] += 1
        seen_full[user] = rid
        kept.append({"id": rid, "messages": [r["messages"][0], r["messages"][1], {"role": "assistant", "content": json.dumps(lab)}]})
    return kept, log, stats
