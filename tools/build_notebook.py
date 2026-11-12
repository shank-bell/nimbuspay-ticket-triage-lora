"""Assemble notebook.ipynb from cell sources + the three modules (embedded with %%writefile so the notebook is self-contained).

    uv run --python 3.11 python -X utf8 tools/build_notebook.py
"""
import ast, json, re, sys

ROOT = "E:/Fine_Tuneing-Assigh/"
cells = []


def md(text):
    cells.append(("markdown", text.strip("\n")))


def code(text):
    cells.append(("code", text.strip("\n")))


def module(path, name):
    cells.append(("code", f"%%writefile /content/nimbus/{name}\n" + open(ROOT + path, encoding="utf-8").read().rstrip("\n")))


# =====================================================================================================================
md(r"""
# NimbusPay ticket triage: fine-tuning Qwen2.5-1.5B-Instruct with LoRA / QLoRA

**Task.** Read a support ticket, return an 8-key JSON triage record. The model gets only a fixed 1-line system prompt + the ticket at test time, so it must *learn* the business rules (priority, null rules, paise conversion, date resolution, id normalisation) from data.

**Setup.** Free Colab **T4** (16 GB, fp16 only, no bf16). Base model `Qwen/Qwen2.5-1.5B-Instruct`. Own training loop (PyTorch + PEFT); no Trainer/TRL.

| Part | What |
|---|---|
| A | Audit and clean `train.jsonl` (rule checker derived from SCHEMA.md) |
| B | Baselines on dev: fixed prompt, and rules-in-prompt |
| C | Pipeline evidence: chat template, loss masking, sequence length, parameter budget |
| D | Controlled experiments (one change at a time) |
| E | Error analysis, final model, test predictions |

**To run:** upload the `candidate_pack` folder to Google Drive at `MyDrive/nimbus/candidate_pack`, select *Runtime > Change runtime type > T4 GPU*, then *Run all*. Adapters, predictions and the results log are saved to `MyDrive/nimbus/work`.

**AI assistance.** An AI coding assistant (an LLM agent) was used to write the audit scripts, this notebook and the training module; every number below is produced by the cells here and was reviewed by me.
""")

code(r"""
# 0. Environment ---------------------------------------------------------------------------------------------------
%pip install -q -U transformers peft accelerate bitsandbytes
# Colab ships torchao 0.10.0 but the current peft refuses to import with torchao < 0.16 (ImportError in get_peft_model).
# We do not use torchao, so remove it.
%pip uninstall -y -q torchao
""")

code(r"""
import os, sys, json, time, gc, random, csv, collections, subprocess
import numpy as np, pandas as pd, torch, transformers, peft
import bitsandbytes as bnb
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__, "| bitsandbytes", bnb.__version__)
print("GPU:", torch.cuda.get_device_name(0), "| memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
if "T4" not in torch.cuda.get_device_name(0):
    print("WARNING: the assignment targets a Colab T4; timings and memory below are for", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount("/content/drive")
PACK = "/content/drive/MyDrive/nimbus/candidate_pack"      # contains data/ and score.py
WORK = "/content/drive/MyDrive/nimbus/work"                # adapters, predictions, results log
for d in ("adapters", "preds", "logs"):
    os.makedirs(f"{WORK}/{d}", exist_ok=True)
os.makedirs("/content/nimbus", exist_ok=True)
assert os.path.exists(f"{PACK}/data/train.jsonl") and os.path.exists(f"{PACK}/score.py"), "upload candidate_pack to MyDrive/nimbus/candidate_pack"
""")

md(r"""
The three modules below are written to disk and imported, so the notebook is self-contained.
`nimbus_lib` = SCHEMA.md rules as code (used to audit and clean); `nimbus_clean` = the cleaning decisions; `nimbus_train` = data -> tensors, loss, training loop, greedy generation.
""")
module("audit/nimbus_lib.py", "nimbus_lib.py")
module("audit/nimbus_clean.py", "nimbus_clean.py")
module("src/nimbus_train.py", "nimbus_train.py")

code(r"""
sys.path.insert(0, "/content/nimbus"); sys.path.insert(0, PACK)
import nimbus_lib as nl, nimbus_clean as nc, nimbus_train as nt, score as scorer
nl.ROOT = f"{PACK}/data/"
raw, dev, test = nl.load("train.jsonl"), nl.load("dev.jsonl"), nl.load("test_inputs.jsonl")
print("rows: train", len(raw), "| dev", len(dev), "| test", len(test))
FIXED_SYSTEM = raw[0]["messages"][0]["content"]
assert all(r["messages"][0]["content"] == FIXED_SYSTEM for r in raw + dev + test)
print("system prompt:", FIXED_SYSTEM)
""")

# =====================================================================================================================
md(r"""
## Part A: audit and clean the training data

`train.jsonl` is not clean. The audit combines (1) a structural profile, (2) sentence-template mining (the tickets are generated from a fixed phrase inventory), (3) a rule checker that re-derives every deterministic field from the ticket text using only SCHEMA.md, judging each field against the *derived truth* rather than against the other label fields, (4) a cross-validated category check, and (5) exact-duplicate / leakage checks. The full scripts and write-up are in `audit/` in the repository; the cells below reproduce the headline numbers and run the cleaning.
""")

code(r"""
# How problems 1-2 were found: structural profile
bad_json = [r["id"] for r in raw if nl.parse_label(r["messages"][2]["content"]) is None]
key_sets = collections.Counter(tuple(k for k in (nl.parse_label(r["messages"][2]["content"]) or {"<unparseable>": 0}))[:2] for r in raw)
print("labels that are not valid JSON (truncated):", len(bad_json), bad_json[:4], "...")
print("label key sets (first 2 keys):", dict(key_sets))
legacy = [r for r in raw if '"prio"' in r["messages"][2]["content"]]
print("legacy v1 rows:", len(legacy), "| example:", legacy[0]["messages"][2]["content"][:140])
""")

code(r"""
clean, log, stats = nc.clean_dataset(raw, dev)
print("input", len(raw), "-> kept", len(clean), "| dropped", sum(a == "drop" for _, a, _ in log), "| fixed", sum(a == "fix" for _, a, _ in log))
print(dict(stats))
assert stats["drop_unexplained"] == 0, "every non-conforming row must match a known corruption signature"
with open(f"{WORK}/cleaning_log.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["id", "action", "reason"]); w.writerows(log)
with open(f"{WORK}/train_clean.jsonl", "w", encoding="utf-8") as f:
    for r in clean:
        f.write(json.dumps(r, ensure_ascii=False) + chr(10))
pd.DataFrame(log, columns=["id", "action", "reason"]).assign(kind=lambda d: d.reason.str.replace(r"[\d]+", "#", regex=True).str[:55]).groupby(["action", "kind"]).size().rename("rows").reset_index()
""")

code(r"""
# Evidence that it is the *priority* column that is corrupted (problem 8): needs_human always follows the spec-derived priority
pri_fixed = [i for i, a, why in log if a == "fix" and "contradicts spec" in why]
by_id = {r["id"]: r for r in raw}
agree_derived = informative = informative_agree = 0
for i in pri_fixed:
    lab = nl.parse_label(by_id[i]["messages"][2]["content"])
    lab = {("priority" if k == "prio" else k): v for k, v in lab.items()}
    d = nl.derive(by_id[i]["messages"][1]["content"], lab["category"])
    agree_derived += lab["needs_human"] == d["needs_human"]
    if nl.exp_needs_human(lab["priority"], d["_contacted"]) != d["needs_human"]:   # rows where the bad priority WOULD imply another needs_human
        informative += 1
        informative_agree += lab["needs_human"] == d["needs_human"]
print(f"priority-fixed rows: {len(pri_fixed)} | needs_human equals the spec-derived value in {agree_derived}/{len(pri_fixed)}")
print(f"in the {informative} rows where the corrupted priority would imply a different needs_human, the label agrees with the derived priority in {informative_agree}/{informative} (and with the corrupted one in {informative - informative_agree})")

# Validation of the checker itself: derive() vs the dev gold labels (dev used for MEASUREMENT only, never to decide a fix)
fields = ["priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
hit = collections.Counter()
for r in dev:
    gold = json.loads(r["messages"][2]["content"]); d = nl.derive(r["messages"][1]["content"], gold["category"])
    for k in fields:
        hit[k] += d[k] == gold[k]
print("derive() vs dev gold:", {k: f"{hit[k] / len(dev):.1%}" for k in fields})
DEV_LEAK = {r["id"] for r in dev if r["messages"][1]["content"] in {x["messages"][1]["content"] for x in raw}}
print("dev tickets that also appear in raw train:", len(DEV_LEAK))
""")

# =====================================================================================================================
md(r"""
## Evaluation helpers
Scoring uses the provided `score.py` untouched. Besides the official dev numbers, every run also reports `exact_clean_dev`: exact match on the 170 dev tickets that do **not** appear in raw train, so the leakage in the raw file cannot flatter a run.
""")

code(r"""
RESULTS = f"{WORK}/logs/results.jsonl"


def evaluate(gold_rows, texts, tag=None):
    preds = [{"id": g["id"], "output": t} for g, t in zip(gold_rows, texts)]
    if tag:
        with open(f"{WORK}/preds/{tag}.jsonl", "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + chr(10))
    res = scorer.score(gold_rows, preds)
    res_clean = scorer.score([g for g in gold_rows if g["id"] not in DEV_LEAK], preds)
    a = res["all"]
    return {"exact": a["exact_match"], "field_acc": a["mean_field_acc"], "json_valid": a["json_valid"],
            "exact_clean_dev": res_clean["all"]["exact_match"], "fields": a["fields"], "by_category": {k: v["exact_match"] for k, v in res.items() if k.startswith("category=")}}


def log_result(entry):
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + chr(10))


def results_table():
    rows = [json.loads(l) for l in open(RESULTS, encoding="utf-8")] if os.path.exists(RESULTS) else []
    last = {r["name"]: r for r in rows if not r["name"].endswith("__train_only")}   # latest entry per run name
    cols = ["name", "desc", "n_train", "trainable_M", "exact", "field_acc", "json_valid", "exact_clean_dev", "exact_rep1.0", "peak_gb", "train_min"]
    df = pd.DataFrame(list(last.values())).reindex(columns=cols)
    return df.round(3)
""")

# =====================================================================================================================
md(r"""
## Part B: baselines on dev (no fine-tuning)

1. **B1:** base model, the fixed system prompt + ticket (exactly what the fine-tuned model will get).
2. **B2:** same model with the full SCHEMA.md rules written into the system prompt (allowed here only, as a reference point).

Greedy decoding, `max_new_tokens=200`, the model's own chat template, strict scoring (the output must be *only* a JSON object). I also print a lenient score (code fences / prose stripped) as a diagnostic, never as the result.
""")

code(r"""
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(nt.MODEL_ID)
print("eos:", tok.eos_token, "| pad:", tok.pad_token, "| padding_side:", tok.padding_side)

torch.cuda.reset_peak_memory_stats()
base = nt.load_base(quant4=False)
print("base model loaded in fp16, params:", round(sum(p.numel() for p in base.parameters()) / 1e9, 2), "B | GPU GB:", round(nt.peak_gb(), 2))
print("model generation_config:", base.generation_config)
""")

code(r"""
# B1: fixed prompt
t0 = time.time(); out_b1 = nt.generate(base, tok, dev, batch_size=16); m1 = (time.time() - t0) / 60
r_b1 = evaluate(dev, out_b1, "B1_base_fixed_prompt_dev")
lenient = evaluate(dev, [nt.lenient_json(t) for t in out_b1])
print(f"B1 strict: exact {r_b1['exact']:.1%} | field acc {r_b1['field_acc']:.1%} | valid JSON {r_b1['json_valid']:.1%}   ({m1:.1f} min)")
print(f"B1 lenient (diagnostic only): exact {lenient['exact']:.1%} | field acc {lenient['field_acc']:.1%} | valid {lenient['json_valid']:.1%}")
for i in (0, 1, 2):
    print("--- raw output", i, "---"); print(out_b1[i][:400])
log_result({"name": "B1_base_fixed_prompt", "desc": "base model, fixed prompt, no fine-tuning", "n_train": 0, "trainable_M": 0, "exact": r_b1["exact"], "field_acc": r_b1["field_acc"],
            "json_valid": r_b1["json_valid"], "exact_clean_dev": r_b1["exact_clean_dev"], "peak_gb": nt.peak_gb(), "train_min": 0})
""")

code(r"""
# B2: rules in the prompt (reference only)
RULES = open(f"{PACK}/SCHEMA.md", encoding="utf-8").read()
SYSTEM_RULES = FIXED_SYSTEM + "\n\n" + RULES + "\n\nReply with the JSON object only, no explanation and no code fences."
t0 = time.time(); out_b2 = nt.generate(base, tok, dev, batch_size=4, system_override=SYSTEM_RULES); m2 = (time.time() - t0) / 60
r_b2 = evaluate(dev, out_b2, "B2_base_rules_in_prompt_dev")
lenient2 = evaluate(dev, [nt.lenient_json(t) for t in out_b2])
print(f"B2 strict: exact {r_b2['exact']:.1%} | field acc {r_b2['field_acc']:.1%} | valid JSON {r_b2['json_valid']:.1%}   ({m2:.1f} min)")
print(f"B2 lenient (diagnostic only): exact {lenient2['exact']:.1%} | field acc {lenient2['field_acc']:.1%} | valid {lenient2['json_valid']:.1%}")
print({k: round(v, 3) for k, v in r_b2["fields"].items()})
print("--- raw output 0 ---"); print(out_b2[0][:400])
log_result({"name": "B2_base_rules_in_prompt", "desc": "base model, SCHEMA.md rules in system prompt (reference)", "n_train": 0, "trainable_M": 0, "exact": r_b2["exact"], "field_acc": r_b2["field_acc"],
            "json_valid": r_b2["json_valid"], "exact_clean_dev": r_b2["exact_clean_dev"], "peak_gb": nt.peak_gb(), "train_min": 0})
del base; nt.free_memory()
""")

# =====================================================================================================================
md(r"""
## Part C: the fine-tuning pipeline

Evidence for each design choice (template, loss tokens, sequence length, parameter budget) is printed below.

**Chat template.** The prompt is built with the model's own template on `[system, user]` plus `add_generation_prompt=True`: this is exactly what `generate` receives at test time. The target is the assistant JSON followed by `<|im_end|>`, so the model learns to stop (the output must be *only* the JSON).

**Loss on the right tokens.** Prompt tokens are masked; the loss is token cross-entropy on the assistant JSON and `<|im_end|>` only. It is computed only at those positions (the 152k-vocabulary logits are never built for prompt tokens).
""")

code(r"""
ex = clean[1]
full = tok.apply_chat_template(ex["messages"], tokenize=False)
prompt = tok.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True)
print(full)
assert full.startswith(prompt), "training prompt must be a prefix of the full chat"
e = nt.build_example(ex, tok)
print("\nprompt tokens:", e["n_prompt"], "| supervised tokens:", e["n_target"])
print("supervised text:", repr(tok.decode([t for t in e["labels"] if t != -100])))
print("masked prompt tokens contribute no loss:", all(l == -100 for l in e["labels"][:e["n_prompt"]]))
def joint_ids(r):   # tokenise prompt + completion as ONE string
    s = tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True) + r["messages"][2]["content"] + tok.eos_token
    return tok(s, add_special_tokens=False)["input_ids"]
joint_mismatch = sum(joint_ids(r) != nt.build_example(r, tok)["input_ids"] for r in clean)
print("rows where separate prompt/completion tokenisation differs from joint tokenisation:", joint_mismatch, "of", len(clean))
print("numbers are tokenised digit by digit:", tok.tokenize('"amount_paise": 4750000'))
""")

code(r"""
# Sequence length chosen from evidence
exs_clean = [nt.build_example(r, tok) for r in clean]
L = np.array([len(e["input_ids"]) for e in exs_clean]); T = np.array([e["n_target"] for e in exs_clean])
print("total tokens per example percentiles:", {p: int(np.percentile(L, p)) for p in (50, 90, 95, 97, 99, 100)})
print("assistant tokens per example (p50/p99/max):", int(np.median(T)), int(np.percentile(T, 99)), int(T.max()))
for cap in (320, 512, 1024, 2048, 2560):
    print(f"  cap {cap:5d}: {int((L > cap).sum()):4d} examples ({(L > cap).mean():.1%}) would be cut")
print("examples longer than 512 tokens all contain a quoted '>' thread:", all(">" in clean[i]["messages"][1]["content"] for i in np.where(L > 512)[0]))
MAX_LEN = 2560
assert L.max() <= MAX_LEN
print("-> MAX_LEN =", MAX_LEN, f"(longest example {L.max()} tokens; nothing truncated)")
""")

md(r"""
**Sequence length.** The distribution is bimodal: 97% of examples are at most ~260 tokens, and the other ~3% are tickets with a quoted earlier-email thread (1,000-2,200 tokens). The label sits at the *end* of the sequence, so any truncation would delete the target. Dropping those tickets would also remove the only examples that teach the spec rule *"dates inside quoted earlier emails are not the transaction date"*. So: no truncation, `MAX_LEN = 2560` (longest example rounded up). Memory is bounded by token-budget micro-batching (<= 4,608 padded tokens per forward pass, similar lengths grouped), gradient checkpointing, and computing logits only on assistant positions.

**Parameter budget.** LoRA on all linear layers (attention q,k,v,o and MLP gate,up,down): the rules need arithmetic/format conversion, which lives in the MLPs as much as in attention. The cap is 50M trainable parameters:
""")

code(r"""
cfg = transformers.AutoConfig.from_pretrained(nt.MODEL_ID)
H, I, NL = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
kv = cfg.num_key_value_heads * (H // cfg.num_attention_heads)
dims = {"q_proj": (H, H), "k_proj": (H, kv), "v_proj": (H, kv), "o_proj": (H, H), "gate_proj": (H, I), "up_proj": (H, I), "down_proj": (I, H)}
for r_ in (8, 16, 32, 64):
    n = NL * sum(r_ * (a + b) for a, b in dims.values())
    print(f"rank {r_:2d}: {n / 1e6:6.2f}M trainable parameters  {'<= 50M OK' if n <= 50e6 else '> 50M NOT ALLOWED'}")
""")

md(r"""
**Default configuration (run R0), with reasons**

| Setting | Value | Why |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct, fp16 | <= 4B; chat template has a system role; fits a T4 in 16-bit *and* 4-bit so the LoRA/QLoRA comparison is fair |
| LoRA | r = 16, alpha = 32, dropout 0.05, all linear layers | 18.5M params (cap 50M); alpha = 2r is the usual scale; r = 64 would break the cap |
| Optimiser | AdamW, lr 2e-4, cosine decay, 5% warmup, no weight decay | standard LoRA learning rate; varied later in Part D |
| Batch | 16 sequences per step (token-mean loss) | ~83 steps/epoch on 1,320 rows |
| Epochs | 3 | ~250 steps; varied later in Part D |
| Precision | fp16 autocast + GradScaler, fp32 LoRA weights | T4 has no bf16; fp32 adapters keep the update stable |
| Memory | gradient checkpointing, micro-batch budget 4,608 tokens | needed for the 2,200-token quoted-thread tickets |
| Decoding | greedy, `max_new_tokens=200`, model's own generation_config untouched | exactly as the assignment specifies |
""")

code(r"""
def run_experiment(name, train_rows, *, desc, quant4=False, r=16, alpha=None, lr=2e-4, epochs=3, dropout=0.05, seed=0, batch_size=16, sens=True):
    nt.free_memory()
    # a model left over from an earlier cell would inflate the peak-memory number and can cause OOM at evaluation time
    assert torch.cuda.memory_allocated() < 0.5 * 2**30, f"{torch.cuda.memory_allocated() / 2**30:.1f} GB still allocated: run  del model; nt.free_memory()  first"
    torch.cuda.reset_peak_memory_stats(); nt.seed_everything(seed)
    base = nt.load_base(quant4=quant4)
    model = nt.add_lora(base, r=r, alpha=alpha, dropout=dropout, quant4=quant4)
    trainable, total = nt.count_params(model)
    exs = [nt.build_example(x, tok) for x in train_rows]
    assert max(len(e["input_ids"]) for e in exs) <= MAX_LEN
    print(f"[{name}] {desc}\n  rows {len(exs)} | trainable {trainable / 1e6:.2f}M of {total / 1e6:.0f}M | quant4={quant4} r={r} lr={lr} epochs={epochs}")
    res = nt.train(model, tok, exs, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed)
    train_min, peak = res["seconds"] / 60, nt.peak_gb()
    model.save_pretrained(f"{WORK}/adapters/{name}")
    log_result({"name": name + "__train_only", "desc": desc, "n_train": len(exs), "trainable_M": trainable / 1e6, "peak_gb": peak, "train_min": train_min})   # survives an evaluation crash
    print(f"  training done: {train_min:.1f} min, peak {peak:.1f} GB, adapter saved")
    t0 = time.time(); outs = nt.generate(model, tok, dev, batch_size=16); eval_min = (time.time() - t0) / 60
    ev = evaluate(dev, outs, f"{name}_dev")
    entry = {"name": name, "desc": desc, "n_train": len(exs), "trainable_M": trainable / 1e6, "quant4": quant4, "r": r, "lr": lr, "epochs": epochs,
             "exact": ev["exact"], "field_acc": ev["field_acc"], "json_valid": ev["json_valid"], "exact_clean_dev": ev["exact_clean_dev"],
             "fields": ev["fields"], "by_category": ev["by_category"], "peak_gb": peak, "train_min": train_min, "eval_min": eval_min,
             "skipped_steps": res["skipped_steps"], "final_loss": float(np.mean([l for _, l in res["loss_hist"][-10:]]))}
    if sens:   # sensitivity: the model's generation_config has repetition_penalty=1.1, which stays active in the default run
        ev2 = evaluate(dev, nt.generate(model, tok, dev, batch_size=16, repetition_penalty=1.0))
        entry["exact_rep1.0"], entry["field_acc_rep1.0"] = ev2["exact"], ev2["field_acc"]
    log_result(entry)
    print(f"  -> exact {ev['exact']:.1%} | field acc {ev['field_acc']:.1%} | valid {ev['json_valid']:.1%} | exact on clean dev {ev['exact_clean_dev']:.1%}"
          + (f" | exact with repetition_penalty=1.0: {entry['exact_rep1.0']:.1%}" if sens else "")
          + f" | peak {peak:.1f} GB | train {train_min:.1f} min")
    return model, entry
""")

md(r"""
### R0: reference run (cleaned data, 16-bit LoRA)
""")

code(r"""
m, e0 = run_experiment("R0_clean_lora16", clean, desc="cleaned data, 16-bit LoRA r=16, lr 2e-4, 3 epochs")
del m; nt.free_memory()   # the adapter is saved to Drive; do not keep the model on the GPU
""")

code(r"""
print({k: round(v, 3) for k, v in e0["fields"].items()})
print("exact match by category:", {k.split("=")[1]: round(v, 2) for k, v in e0["by_category"].items()})
results_table()
""")

# =====================================================================================================================
md(r"""
## Part D: controlled experiments

Every run below changes **exactly one thing** relative to R0 (cleaned data, 16-bit LoRA r=16, lr 2e-4, 3 epochs, seed 0) and is scored on the same 200 dev tickets. Each run logs exact match, mean field accuracy, peak GPU memory (training), training minutes, and exact match with `repetition_penalty=1.0` (the model's default generation_config applies 1.1).

**How this section runs.** Re-running all experiments takes about 2 hours on a T4, which would break the 3-hour limit for the whole notebook. So by default (`RUN_EXPERIMENTS = False`) the experiment cells are skipped and the table is rebuilt from the log of my original Colab runs (embedded below, produced by exactly the code in these cells). Set `RUN_EXPERIMENTS = True` to re-run everything. R0 above and the final model below are always run live, which also checks that the saved R0 numbers reproduce.
""")

_log_path = ROOT + "experiment_log.jsonl"
_saved = open(_log_path, encoding="utf-8").read() if __import__("os").path.exists(_log_path) else ""
_nl, _q = chr(10), chr(34) * 3
cells.append(("code", "# Log of my original Part D runs (Colab T4, seed 0): one JSON line per run, written by run_experiment()" + _nl
              + "SAVED_LOG = r" + _q + _nl + _saved.strip() + _nl + _q))

code(r"""
RUN_EXPERIMENTS = False     # True: re-run every Part D experiment (about 2 h on a T4)

def saved_table():
    rows = [json.loads(l) for l in SAVED_LOG.strip().splitlines()]
    cols = ["name", "desc", "n_train", "trainable_M", "exact", "field_acc", "json_valid", "exact_clean_dev", "exact_rep1.0", "peak_gb", "train_min"]
    return pd.DataFrame(rows).reindex(columns=cols).round(3)

def run_or_skip(name, rows, **kw):
    if RUN_EXPERIMENTS:
        m, e = run_experiment(name, rows, **kw)
        del m; nt.free_memory()
        return e
    print(f"RUN_EXPERIMENTS=False: {name} not re-run, its result is in the saved log table")

saved_table()
""")

md(r"""
### D1: raw vs cleaned data
*Hypothesis:* the raw file teaches the model the corruptions (rupees-for-paise amounts, un-normalised ids, flipped priorities, truncated JSON, v1 schema), so exact match on the 170 dev tickets that are *not* in raw train should drop. The 30 leaked dev tickets can inflate the raw run's official dev number, which is why `exact_clean_dev` is reported too.
""")

code(r"""
e_d1 = run_or_skip("D1_raw_lora16", raw, desc="RAW train.jsonl (1,445 rows, uncleaned), otherwise as R0")
""")

md(r"""
### D2: 16-bit LoRA vs 4-bit QLoRA
*Hypothesis:* NF4 quantisation of the frozen base saves roughly 1.5-2 GB of GPU memory but costs speed (dequantisation) and perhaps a little accuracy. Everything else equals R0.
""")

code(r"""
e_d2 = run_or_skip("D2_clean_qlora4", clean, quant4=True, desc="cleaned data, 4-bit QLoRA (NF4, double quant, fp16 compute), otherwise as R0")
""")

md(r"""
### D3: number of epochs
*Hypothesis:* the training loss is already ~0.03 after one epoch (R0 log above), so one epoch may be enough and 5 epochs may over-fit the templates and hurt paraphrase robustness. Everything else equals R0.
""")

code(r"""
e_d3a = run_or_skip("D3a_clean_lora16_1epoch", clean, epochs=1, desc="cleaned data, 16-bit LoRA, 1 epoch (R0: 3)")
e_d3b = run_or_skip("D3b_clean_lora16_5epochs", clean, epochs=5, desc="cleaned data, 16-bit LoRA, 5 epochs (R0: 3)")
""")

md(r"""
### D4 (my own idea): emit the derived fields last
The scorer parses the JSON, so key order does not matter for scoring, but it matters for the model: a language model decides each token from the tokens before it. In the spec's order, `priority` and `needs_human` are generated **before** `amount_paise`, so the model must evaluate "amount >= Rs 5,000 / 50,000" implicitly. `priority` is the weakest field in R0 (92%). If the derived fields come *last*, `priority` can attend to the category and the amount digits already written, and `needs_human` can attend to `priority`.

*Change:* only the key order of the training targets: `category, amount_paise, txn_date, txn_id, channel, language, priority, needs_human`. Same 8 keys, same values, same data and hyperparameters. The output is still only the JSON object and no post-processing is involved.
""")

code(r"""
ORDER = ["category", "amount_paise", "txn_date", "txn_id", "channel", "language", "priority", "needs_human"]
assert sorted(ORDER) == sorted(nl.KEYS)
def reorder(rows, order=ORDER):
    out = []
    for r in rows:
        lab = json.loads(r["messages"][2]["content"])
        out.append({"id": r["id"], "messages": [r["messages"][0], r["messages"][1], {"role": "assistant", "content": json.dumps({k: lab[k] for k in order})}]})
    return out
clean_reordered = reorder(clean)
print(clean[3]["messages"][2]["content"]); print(clean_reordered[3]["messages"][2]["content"])
e_d4 = run_or_skip("D4_clean_lora16_keyorder", clean_reordered, desc="cleaned data, derived fields (priority, needs_human) emitted last, otherwise as R0")
""")

md(r"""
### Seed noise
Dev has only 200 tickets, so one ticket is 0.5 points and differences of ~2 points are within noise. To calibrate, R0 is repeated with a different seed (different shuffling and LoRA initialisation).
""")

code(r"""
e_seed = run_or_skip("R0b_clean_lora16_seed1", clean, seed=1, desc="R0 repeated with seed 1 (seed noise)")
saved_table()
""")

md(r"""
### Part D: findings
All runs: Colab T4, seed 0, the same 200 dev tickets (one ticket = 0.5 points; the 95% interval of a binomial proportion near 90% on 200 tickets is about +-4 points, so single-seed differences of 1-2 points are not significant).

| Experiment | Result (exact match) | What it shows |
|---|---|---|
| **D1** raw vs cleaned data | 79.0% vs 89.0% (clean-dev 77.6% vs 88.8%) | Cleaning is worth about **+10 points**: this effect is far larger than the noise. The raw-trained model is also fragile: with `repetition_penalty=1.0` it jumps to 87.0% (+8 points), the clean model gains only +1.5. |
| **D2** 16-bit LoRA vs 4-bit QLoRA | 89.0% vs 88.0% | Within noise (2 tickets). QLoRA peaked at 4.7 GB vs 5.3 GB (-0.6 GB) and trained 24% slower (17.7 vs 14.3 min). The memory saving is small, probably because the peak is dominated by activations and the fp32 upcast of the non-quantised layers, not by the weights (not measured separately). |
| **D3** epochs | 1 epoch 42.0%, 3 epochs 89.0%, 5 epochs **93.0%** (clean-dev 92.4%) | My hypothesis (one epoch is enough, more epochs over-fit) was **wrong**. Training loss after 1 epoch (0.024) looks small, but exact match needs every digit right: the per-example loss must be near zero, and it only gets there after 3-5 epochs (final loss 0.0240 / 0.0013 / 0.0002). 5 epochs also removes the repetition-penalty sensitivity (93.0% with and without it). |
| **D4** derived fields last | 87.0% vs 89.0% | **Negative result.** Emitting `priority` and `needs_human` after the amount did not help (-2 points, 4 tickets, within noise). So the priority errors are not caused by the model having to decide before it has written the amount; see the error analysis. |

The final model uses **5 epochs** by a rule fixed before D3b finished (5 epochs only if its clean-dev exact match beats R0's by at least 2 points). Seed-to-seed noise was measured separately on other hardware (see the report; non-T4).
""")

# =====================================================================================================================
md(r"""
## Part E: error analysis of the best model on dev
""")

code(r"""
FINAL_EPOCHS = 5            # decided by a rule fixed before seeing D3b: 5 epochs only if D3b's exact match on the 170 clean dev tickets beats R0's by >= 2 points (92.4% vs 88.8% -> yes)
if FINAL_EPOCHS == 3:
    BEST = "R0_clean_lora16"          # the live R0 run above is the final model
else:
    m, e_final = run_experiment(f"FINAL_clean_lora16_{FINAL_EPOCHS}ep", clean, epochs=FINAL_EPOCHS, desc=f"final model: cleaned data, 16-bit LoRA, {FINAL_EPOCHS} epochs")
    del m; nt.free_memory()
    BEST = f"FINAL_clean_lora16_{FINAL_EPOCHS}ep"
print("final model:", BEST)

def dev_errors(tag):
    preds = {p["id"]: p["output"] for p in map(json.loads, open(f"{WORK}/preds/{tag}_dev.jsonl", encoding="utf-8"))}
    errs = []
    for g in dev:
        gold = json.loads(g["messages"][2]["content"])
        try:
            pr = json.loads(preds[g["id"]])
        except Exception:
            pr = None
        bad = [k for k in nl.KEYS if pr is None or k not in pr or pr[k] != gold[k] or type(pr[k]) is not type(gold[k])]
        if bad:
            errs.append({"id": g["id"], "bad": bad, "gold": gold, "pred": pr, "text": g["messages"][1]["content"].split("\n\n", 1)[1], "raw": preds[g["id"]]})
    return errs

errs = dev_errors(BEST)
print(f"{BEST}: {len(errs)} of {len(dev)} dev tickets have at least one wrong field")
print("wrong fields:", collections.Counter(k for e in errs for k in e["bad"]).most_common())
print("tickets by number of wrong fields:", dict(collections.Counter(len(e["bad"]) for e in errs)))
pri = collections.Counter((e["gold"]["priority"], (e["pred"] or {}).get("priority")) for e in errs if "priority" in e["bad"])
print("priority errors (gold -> predicted):", dict(pri))
cat = collections.Counter((e["gold"]["category"], (e["pred"] or {}).get("category")) for e in errs if "category" in e["bad"])
print("category errors (gold -> predicted):", dict(cat))
""")

code(r"""
def show(e, width=420):
    print(f"--- {e['id']}  wrong: {e['bad']}")
    print("   ticket:", e["text"].replace(chr(10), " / ")[:width])
    for k in e["bad"]:
        print(f"   {k:13} gold={e['gold'][k]!r:>16}   pred={(e['pred'] or {}).get(k, '<missing>')!r}")
for e in errs[:14]:
    show(e)

def paired(tag_a, tag_b):
    A = {e["id"] for e in dev_errors(tag_a)}; B = {e["id"] for e in dev_errors(tag_b)}
    print(f"{tag_a}: {len(A)} tickets wrong | {tag_b}: {len(B)} wrong | only {tag_a} wrong: {len(A - B)} | only {tag_b} wrong: {len(B - A)} | both wrong: {len(A & B)} | both right: {len(dev) - len(A | B)}")
if BEST != "R0_clean_lora16":
    paired("R0_clean_lora16", BEST)      # paired view of the same 200 tickets: how many flip each way
""")

# =====================================================================================================================
md(r"""
## Final model: predictions for dev and test
The final adapter is reloaded from disk exactly as the graders will do it (base model in fp16 + PEFT adapter), then decoded with plain greedy `generate`, `max_new_tokens=200`, one ticket at a time (no padding), the model's own chat template and generation_config, and only `.strip()` applied to the text.
""")

code(r"""
from peft import PeftModel
nt.free_memory(); torch.cuda.reset_peak_memory_stats()
base = nt.load_base(quant4=False)
final = PeftModel.from_pretrained(base, f"{WORK}/adapters/{BEST}")
trainable_final = sum(p.numel() for p in final.parameters() if p.requires_grad)
print("adapter loaded from", f"{WORK}/adapters/{BEST}", "| files:", sorted(os.listdir(f"{WORK}/adapters/{BEST}")))
print("adapter parameters:", sum(p.numel() for n, p in final.named_parameters() if "lora_" in n) / 1e6, "M")
t0 = time.time(); dev_out1 = nt.generate(final, tok, dev, batch_size=1); print(f"dev, batch size 1: {(time.time() - t0) / 60:.1f} min")
ev = evaluate(dev, dev_out1, f"{BEST}_final_dev_bs1")
print(f"dev (batch size 1): exact {ev['exact']:.1%} | field acc {ev['field_acc']:.1%} | valid {ev['json_valid']:.1%}")
""")

code(r"""
t0 = time.time(); test_out = nt.generate(final, tok, test, batch_size=1); print(f"test, batch size 1: {(time.time() - t0) / 60:.1f} min")
with open(f"{WORK}/predictions_test.jsonl", "w", encoding="utf-8") as f:
    for r, o in zip(test, test_out):
        f.write(json.dumps({"id": r["id"], "output": o}, ensure_ascii=False) + chr(10))
valid = sum(nl.parse_label(o) is not None and set(nl.parse_label(o)) == set(nl.KEYS) for o in test_out)
print("test predictions written:", len(test_out), "| valid 8-key JSON objects:", valid)
print(test_out[0])
""")

md(r"""
**Robustness checks of the submitted predictions.** (1) Does batching change the greedy output? (2) How much does the model's default `repetition_penalty=1.1` change test outputs compared with 1.0? The submitted file uses the default generation_config, i.e. what plain `model.generate(..., do_sample=False)` does.
""")

code(r"""
t0 = time.time()
test_b16 = nt.generate(final, tok, test, batch_size=16)
test_rp1 = nt.generate(final, tok, test, batch_size=16, repetition_penalty=1.0)
print(f"({(time.time() - t0) / 60:.1f} min)")
print("identical to the submitted file: batch size 16 vs 1:", sum(a == b for a, b in zip(test_out, test_b16)), "/", len(test))
print("identical to the submitted file: repetition_penalty 1.0 vs default 1.1:", sum(a == b for a, b in zip(test_out, test_rp1)), "/", len(test))
""")

# =====================================================================================================================
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
out = ROOT + "notebook.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# ---- syntax check every code cell (magics removed)
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
