"""Part A, step 1: structural profile of train.jsonl (no schema rules applied yet)."""
import json, re, sys
from collections import Counter

ROOT = "E:/Fine_Tuneing-Assigh/training_assessment/candidate_pack/data/"
KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]


def load(name):
    with open(ROOT + name, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


rows = load("train.jsonl")
print("rows:", len(rows))
print("row keys:", Counter(tuple(sorted(r)) for r in rows))
print("n messages:", Counter(len(r["messages"]) for r in rows))
print("roles:", Counter(tuple(m["role"] for m in r["messages"]) for r in rows))
print("system prompts:", Counter(r["messages"][0]["content"] for r in rows))

ids = Counter(r["id"] for r in rows)
print("duplicate ids:", [i for i, c in ids.items() if c > 1][:10])

bad_json, keysets, types = [], Counter(), {k: Counter() for k in KEYS}
vals = {k: Counter() for k in KEYS}
for r in rows:
    try:
        lab = json.loads(r["messages"][-1]["content"])
    except Exception as e:
        bad_json.append((r["id"], r["messages"][-1]["content"][:100]))
        continue
    if not isinstance(lab, dict):
        bad_json.append((r["id"], "not dict"))
        continue
    keysets[tuple(lab.keys())] += 1
    for k in KEYS:
        if k in lab:
            types[k][type(lab[k]).__name__] += 1
            if k in ("category", "priority", "channel", "language", "needs_human"):
                vals[k][repr(lab[k])] += 1
print("\nunparseable labels:", len(bad_json), bad_json[:5])
print("\nkey sets:")
for ks, c in keysets.most_common():
    print(" ", c, list(ks))
print("\ntypes per field:")
for k in KEYS:
    print(" ", k, dict(types[k]))
print("\nvalue counts:")
for k, c in vals.items():
    print(" ", k, dict(c))

# user message header format
hdr = re.compile(r"^Received: (\d{4}-\d{2}-\d{2}) \((Mon|Tue|Wed|Thu|Fri|Sat|Sun)\)\nSource: (app-chat|email|twitter|ivr-transcript)\n\n", re.S)
nohdr = [r["id"] for r in rows if not hdr.match(r["messages"][1]["content"])]
print("\nuser messages with non-standard header:", len(nohdr), nohdr[:10])
print("sources:", Counter(re.search(r"Source: (.*)", r["messages"][1]["content"]).group(1) for r in rows if "Source:" in r["messages"][1]["content"]))
lens = sorted(len(r["messages"][1]["content"]) for r in rows)
print("user chars min/median/max:", lens[0], lens[len(lens)//2], lens[-1])
print("id prefixes:", Counter(r["id"].split("-")[0] for r in rows))
