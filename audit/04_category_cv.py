"""Part A, step 4: find suspicious *category* labels with a cross-validated text classifier (train data only)."""
import re, sys, json, collections
import numpy as np
sys.path.insert(0, "E:/Fine_Tuneing-Assigh/audit")
from nimbus_lib import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline, make_union

rows = load("train.jsonl")
JUNK = re.compile(r"^\s*(test|asdf asdf|test test \d+)?\s*$", re.I)
ids, texts, cats = [], [], []
for r in rows:
    lab = parse_label(r["messages"][2]["content"])
    if not lab or "category" not in lab: continue
    rec, src, body, own = parse_ticket(r["messages"][1]["content"])
    if JUNK.match(own): continue
    t = re.sub(r"\d+", "#", own.lower())
    t = re.sub(r"regards,.*", "", t, flags=re.S)
    ids.append(r["id"]); texts.append(t); cats.append(lab["category"])
print("rows used:", len(ids), collections.Counter(cats))
y = np.array(cats)
vec = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                 TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True))
clf = make_pipeline(vec, LogisticRegression(C=20, max_iter=2000))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
P = cross_val_predict(clf, texts, y, cv=skf, method="predict_proba")
classes = sorted(set(cats))
pred = np.array(classes)[P.argmax(1)]
conf_label = np.array([P[i, classes.index(cats[i])] for i in range(len(ids))])
print("CV accuracy vs labels: %.3f" % (pred == y).mean())
print(collections.Counter(zip(y[pred != y], pred[pred != y])).most_common(25))
sus = [(conf_label[i], ids[i], cats[i], pred[i], P[i].max()) for i in range(len(ids)) if pred[i] != y[i]]
sus.sort()
out = {"suspects": [{"id": i, "label": l, "pred": p, "p_label": float(pl), "p_pred": float(pp)} for pl, i, l, p, pp in sus]}
json.dump(out, open("E:/Fine_Tuneing-Assigh/audit/category_suspects.json", "w"), indent=1)
print("suspects:", len(sus))
