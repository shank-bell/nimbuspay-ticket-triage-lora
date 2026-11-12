"""Part A, step 6: validate the cleaning against independent evidence.

 (1) the rule checker finds no violations in the cleaned file
 (2) for train rows that are copies of dev tickets: cleaned label vs dev gold label (dev labels used for measurement only)
 (3) how well derive() reproduces dev gold labels, per field (given the gold category)
"""
import importlib, json, sys, collections
sys.path.insert(0, "E:/Fine_Tuneing-Assigh/audit")
from nimbus_lib import *

# ---- (2) fix-logic check on the 30 leaked rows
clean = importlib.import_module("nimbus_clean")
dev = {r["messages"][1]["content"]: r for r in load("dev.jsonl")}
tr = load("train.jsonl")
n = same = 0; diffs = collections.Counter(); raw_same = 0; shown = 0
for r in tr:
    u = r["messages"][1]["content"]
    if u not in dev: continue
    gold = json.loads(dev[u]["messages"][2]["content"])
    raw = parse_label(r["messages"][2]["content"])
    if raw == gold: raw_same += 1
    n += 1
    if raw is None:
        lab, notes = None, ["truncated"]
    else:
        lab, notes = clean.clean_row(r)
    if lab == gold: same += 1
    else:
        for k in KEYS:
            if lab is None or lab.get(k) != gold[k]: diffs[k] += 1
        if shown < 3:
            print("  not equal:", r["id"], notes, "\n    cleaned:", lab, "\n    gold:   ", gold); shown += 1
print(f"(2) train rows identical to a dev ticket: {n}; raw label == dev gold: {raw_same}; cleaned label == dev gold: {same}; per-field diffs: {dict(diffs)}")

# ---- (3) derive() vs dev gold
dv = load("dev.jsonl")
hit = collections.Counter(); amb = 0; bad = collections.defaultdict(list)
for r in dv:
    gold = json.loads(r["messages"][2]["content"])
    d = derive(r["messages"][1]["content"], gold["category"])
    if d["ambiguous"]: amb += 1
    for k in KEYS[1:]:
        if d[k] == gold[k]: hit[k] += 1
        else: bad[k].append(r["id"])
print(f"(3) derive() vs dev gold on {len(dv)} dev rows (ambiguous: {amb}):")
for k in KEYS[1:]: print(f"    {k:13} {hit[k]/len(dv):6.1%}   mismatches: {bad[k][:6]}")
