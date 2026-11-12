"""Assemble seeds_notebook.ipynb: re-train the FINAL configuration with seeds 1 and 2 on a Colab T4 (dev scores only).

It does not touch the submitted adapter or predictions. It reuses the same modules as notebook.ipynb (embedded with
%%writefile) and the job runner college/run_college.py, with two extra jobs registered at run time.

    uv run --python 3.11 python -X utf8 tools/build_seeds_notebook.py
"""
import ast, json, sys

ROOT = "E:/Fine_Tuneing-Assigh/"
cells = []


def md(text):
    cells.append(("markdown", text.strip("\n")))


def code(text):
    cells.append(("code", text.strip("\n")))


def module(path, target):
    cells.append(("code", f"%%writefile {target}\n" + open(ROOT + path, encoding="utf-8").read().rstrip("\n")))


md(r"""
# Seed check of the final configuration (Colab T4)

The submitted adapter was trained once (seed 0): clean data, 16-bit LoRA r=16, 5 epochs, lr 2e-4. This notebook trains the **same configuration with seeds 1 and 2** and scores them on the 200 dev tickets with the assignment's `score.py`, so the report can state a spread instead of a single number. It does **not** change the submitted adapter or `predictions_test.jsonl`, and it never looks at test inputs.

Run: Runtime > Change runtime type > T4 GPU, then *Run all* (about 1 hour). Results are appended to `MyDrive/nimbus/seeds_out/results.jsonl`.
""")

code(r"""
# 0. Environment: the same library versions as the final run of notebook.ipynb (2026-11-12) ---------------------------
%pip install -q transformers==5.22.2 peft==0.21.2 accelerate bitsandbytes==0.50.2
%pip uninstall -y -q torchao
""")

code(r"""
import os, sys, json, math, torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
print("GPU:", torch.cuda.get_device_name(0), "| memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
assert "T4" in torch.cuda.get_device_name(0), "this check is meant for a Colab T4"
from google.colab import drive
drive.mount("/content/drive")
PACK = "/content/drive/MyDrive/nimbus/candidate_pack"      # contains data/ and score.py
OUT = "/content/drive/MyDrive/nimbus/seeds_out"            # results.jsonl, dev predictions, adapters of the seed runs
assert os.path.exists(f"{PACK}/data/train.jsonl") and os.path.exists(f"{PACK}/score.py"), "candidate_pack not found on Drive"
for d in ("src", "audit", "college"):
    os.makedirs(f"/content/seeds_code/{d}", exist_ok=True)
os.makedirs(OUT, exist_ok=True)
""")

module("src/nimbus_train.py", "/content/seeds_code/src/nimbus_train.py")
module("audit/nimbus_lib.py", "/content/seeds_code/audit/nimbus_lib.py")
module("audit/nimbus_clean.py", "/content/seeds_code/audit/nimbus_clean.py")
module("college/run_college.py", "/content/seeds_code/college/run_college.py")

code(r"""
# Two extra jobs = the FINAL configuration (clean data, 16-bit LoRA r=16, 5 epochs) with seeds 1 and 2. Everything else is the
# runner's default (lr 2e-4, dropout 0.05, batch 16, MAX_LEN 2560), which is what the final notebook used for seed 0.
sys.path.insert(0, "/content/seeds_code/college")
import run_college as rc
rc.JOBS["FINAL_5ep_seed1"] = dict(desc="final configuration (5 epochs), seed 1", epochs=5, seed=1)
rc.JOBS["FINAL_5ep_seed2"] = dict(desc="final configuration (5 epochs), seed 2", epochs=5, seed=2)
sys.argv = ["run_college.py", "--pack", PACK, "--out", OUT, "--jobs", "FINAL_5ep_seed1,FINAL_5ep_seed2"]
rc.main()                      # about 28 minutes per seed on a T4; each finished seed is appended to results.jsonl at once
""")

code(r"""
# Summary: seed 0 is the final notebook's run (copied from its results.jsonl), seeds 1 and 2 come from this notebook.
rows = [json.loads(l) for l in open(f"{OUT}/results.jsonl", encoding="utf-8") if l.strip()]
runs = {"seed 0 (final notebook)": dict(exact=0.94, field_acc=0.98875, exact_clean_dev=0.9352941176470588, **{"exact_rep1.0": 0.94}, train_min=24.099143981933594, peak_gb=5.379127502441406)}
for r in rows:
    if r["name"].startswith("FINAL_5ep_seed") and "exact" in r:
        runs["seed " + r["name"][-1] + " (this notebook)"] = r
print(f"{'run':26s} {'exact':>7s} {'field':>7s} {'clean-dev':>10s} {'rep 1.0':>8s} {'peak GB':>8s} {'train min':>10s}")
for k, r in runs.items():
    print(f"{k:26s} {r['exact']:7.1%} {r['field_acc']:7.1%} {r['exact_clean_dev']:10.1%} {r['exact_rep1.0']:8.1%} {r['peak_gb']:8.2f} {r['train_min']:10.1f}")
ex = [r["exact"] * 100 for r in runs.values()]
fa = [r["field_acc"] * 100 for r in runs.values()]
def sd(x): return math.sqrt(sum((v - sum(x) / len(x)) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else float("nan")
print(f"\nexact over {len(ex)} seeds: mean {sum(ex)/len(ex):.2f}, sd {sd(ex):.2f}, min {min(ex):.1f}, max {max(ex):.1f}")
print(f"field accuracy over {len(fa)} seeds: mean {sum(fa)/len(fa):.2f}, sd {sd(fa):.2f}, min {min(fa):.2f}, max {max(fa):.2f}")
errors = [r for r in rows if "error" in r]
print("failed jobs:", [(r['name'], r['error']) for r in errors] if errors else "none")
""")

# ---------------------------------------------------------------------------------------------------------------------
nb_cells = []
for kind, text in cells:
    src = text.split("\n")
    src = [l + "\n" for l in src[:-1]] + [src[-1]]
    if kind == "markdown":
        nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
    else:
        nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})
nb = {"cells": nb_cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 4}
out = ROOT + "seeds_notebook.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

bad = 0
for i, (kind, text) in enumerate(cells):
    if kind != "code":
        continue
    body = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith(("%", "!")))
    if text.startswith("%%writefile"):
        body = "\n".join(text.split("\n")[1:])
    try:
        ast.parse(body)
    except SyntaxError as e:
        bad += 1
        print(f"SYNTAX ERROR in cell {i}: {e}")
print(f"wrote {out}: {len(cells)} cells ({sum(k == 'code' for k, _ in cells)} code), syntax errors: {bad}")
sys.exit(1 if bad else 0)
