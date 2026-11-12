"""Part A, step 5: clean train.jsonl -> audit/train_clean.jsonl + audit/cleaning_log.csv (logic in nimbus_clean.py)."""
import csv, json, sys
sys.path.insert(0, "E:/Fine_Tuneing-Assigh/audit")
from nimbus_lib import load
from nimbus_clean import clean_dataset

OUT = "E:/Fine_Tuneing-Assigh/audit/"
rows, dev = load("train.jsonl"), load("dev.jsonl")
kept, log, stats = clean_dataset(rows, dev)
with open(OUT + "train_clean.jsonl", "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + chr(10))
with open(OUT + "cleaning_log.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "action", "reason"])
    w.writerows(log)
print(f"input rows: {len(rows)}  kept: {len(kept)}  dropped: {sum(1 for l in log if l[1]=='drop')}  rows fixed: {sum(1 for l in log if l[1]=='fix')}")
print(dict(stats))
