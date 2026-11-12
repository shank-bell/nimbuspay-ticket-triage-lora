"""Execute the CPU-only cells of notebook.ipynb locally (no Colab, no GPU) to catch NameErrors/typos before burning GPU time."""
import json, sys, os, tempfile, collections, csv, time, gc, random
import numpy as np, pandas as pd, torch, transformers
from transformers import AutoTokenizer

PACK = "E:/Fine_Tuneing-Assigh/training_assessment/candidate_pack"
WORK = tempfile.mkdtemp()
for d in ("adapters", "preds", "logs"): os.makedirs(f"{WORK}/{d}")
sys.path[:0] = ["E:/Fine_Tuneing-Assigh/audit", "E:/Fine_Tuneing-Assigh/src", PACK]
nb = json.load(open("E:/Fine_Tuneing-Assigh/notebook.ipynb", encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
ns = dict(os=os, sys=sys, json=json, time=time, gc=gc, random=random, csv=csv, collections=collections, np=np, pd=pd, torch=torch,
          transformers=transformers, PACK=PACK, WORK=WORK)
SKIP = ("%%writefile", "%pip", "drive.mount", "nt.load_base", "m, e0 = run_experiment", "tok = AutoTokenizer", "nt.generate(base", "del base", "e0[", "errs = dev_errors", "for e in errs", "nt.generate(final", "nt.generate(test", "test_out")
ran = skipped = 0
for i, c in enumerate(cells):
    if any(s in c for s in SKIP):
        skipped += 1
        if "tok = AutoTokenizer" in c:       # the real tokenizer, from the local copy
            ns["tok"] = AutoTokenizer.from_pretrained("E:/Fine_Tuneing-Assigh/scratch_model")
        continue
    print(f"\n===== cell {i} " + "=" * 60)
    exec(compile(c, f"cell{i}", "exec"), ns)
    if "nl.ROOT" in c:
        ns["nt"].MODEL_ID = "E:/Fine_Tuneing-Assigh/scratch_model"      # AutoConfig for the parameter-budget cell
    ran += 1
print(f"\nran {ran} cells, skipped {skipped} GPU/Colab-only cells")

# ---- exercise the evaluation helpers with synthetic predictions
dev, evaluate, log_result, results_table = ns["dev"], ns["evaluate"], ns["log_result"], ns["results_table"]
perfect = [r["messages"][2]["content"] for r in dev]
ev = evaluate(dev, perfect, "dry_perfect_dev"); assert ev["exact"] == 1.0 and ev["exact_clean_dev"] == 1.0 and ev["json_valid"] == 1.0
broken = perfect[:]; broken[0] = "```json\n" + broken[0] + "\n```"; broken[1] = ""
ev2 = evaluate(dev, broken); assert abs(ev2["exact"] - 198 / 200) < 1e-9 and abs(ev2["json_valid"] - 198 / 200) < 1e-9
log_result({"name": "X", "desc": "d", "n_train": 1, "trainable_M": 1.0, "exact": 0.5, "field_acc": 0.9, "json_valid": 1.0, "exact_clean_dev": 0.4, "peak_gb": 1.0, "train_min": 1.0})
print(results_table().to_string())
print("DRY RUN OK")
