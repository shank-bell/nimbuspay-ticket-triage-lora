"""Part A, step 2: mine the sentence templates used in ticket bodies (digits -> #)."""
import json, re, sys
from collections import Counter, defaultdict

ROOT = "E:/Fine_Tuneing-Assigh/training_assessment/candidate_pack/data/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "templates.txt"


def load(name):
    with open(ROOT + name, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def body_of(user):
    # drop the 2 header lines + blank line
    return user.split("\n\n", 1)[1]


def sentences(body):
    lines = [l for l in body.split("\n") if not l.startswith(">")]
    text = " ".join(l.strip() for l in lines if l.strip())
    # split on sentence enders but keep things like "Rs.13027/-" together
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[@#\u20b9])", text)
    return [p.strip() for p in parts if p.strip()]


def norm(s):
    return re.sub(r"\d", "#", s)


rows = load("train.jsonl")
tmpl = Counter()
example = {}
for r in rows:
    for s in sentences(body_of(r["messages"][1]["content"])):
        t = norm(s)
        tmpl[t] += 1
        example.setdefault(t, r["id"])

print("unique sentence templates:", len(tmpl), "of", sum(tmpl.values()), "sentences")
with open(OUT, "w", encoding="utf-8") as f:
    for t, c in tmpl.most_common():
        f.write(f"{c:5d} | {t}\n")
print("wrote", OUT)
