"""Run fine-tuning experiments on a machine that is NOT the Colab T4 (the college PC) and package the results.

Everything this script produces is NON-T4: say so in the report. Memory and time figures from here are not comparable
with the T4 table, and the final submitted adapter must still be reproduced on a fresh Colab T4.

    python run_college.py --pack ../candidate_pack --jobs R0_repeat,R0b_seed1,D3b_5epochs,D4_keyorder

Output: <out>/college_results.zip  (adapters + dev predictions + results.jsonl + environment). It contains NO company data.
"""
import argparse, collections, csv, json, os, platform, sys, time, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
for base in (HERE, os.path.dirname(HERE)):          # works from the zip layout and from the repo layout
    for sub in ("src", "audit"):
        p = os.path.join(base, sub)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

# name -> overrides of the R0 configuration (clean data, 16-bit LoRA r=16, lr 2e-4, 3 epochs, seed 0)
JOBS = {
    "R0_repeat":    dict(desc="R0 configuration repeated on this machine (hardware check)"),
    "R0b_seed1":    dict(desc="R0 with seed 1 (seed noise)", seed=1),
    "D3a_1epoch":   dict(desc="1 epoch", epochs=1),
    "D3b_5epochs":  dict(desc="5 epochs", epochs=5),
    "D4_keyorder":  dict(desc="derived fields (priority, needs_human) emitted last", reorder=True),
    "D5_rank8":     dict(desc="LoRA rank 8", r=8),
    "D5_rank32":    dict(desc="LoRA rank 32 (36.9M params)", r=32),
    "D6_lr1e-4":    dict(desc="learning rate 1e-4", lr=1e-4),
    "D6_lr4e-4":    dict(desc="learning rate 4e-4", lr=4e-4),
    "D2_qlora4":    dict(desc="4-bit QLoRA", quant4=True),
    # FALLBACK ONLY (T4 unavailable): the final configuration (clean data, 16-bit LoRA r=16, 5 epochs) + test predictions via --test.
    "FINAL_5ep":    dict(desc="final configuration: clean data, 16-bit LoRA r=16, 5 epochs (non-T4 fallback)", epochs=5),
}
DEFAULTS = dict(quant4=False, r=16, lr=2e-4, epochs=3, dropout=0.05, seed=0, reorder=False)
ORDER = ["category", "amount_paise", "txn_date", "txn_id", "channel", "language", "priority", "needs_human"]


def reorder(rows):
    out = []
    for r in rows:
        lab = json.loads(r["messages"][2]["content"])
        out.append({"id": r["id"], "messages": [r["messages"][0], r["messages"][1],
                    {"role": "assistant", "content": json.dumps({k: lab[k] for k in ORDER})}]})
    return out


def tiny_base():
    """Smoke-test model (random weights, CPU). Only used with --smoke."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="folder with data/train.jsonl, data/dev.jsonl, score.py, SCHEMA.md")
    ap.add_argument("--out", default="college_out")
    ap.add_argument("--jobs", default="R0_repeat,R0b_seed1,D3b_5epochs,D4_keyorder", help="comma list from: " + ", ".join(JOBS))
    ap.add_argument("--model", default=None, help="HF id or a local folder (default Qwen/Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--smoke", action="store_true", help="CPU smoke test with a tiny random model")
    ap.add_argument("--test", default=None, help="optional path to test_inputs.jsonl: also write test predictions (greedy, batch size 1) per job")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-dev", type=int, default=0)
    a = ap.parse_args()

    import numpy as np, torch, transformers
    import nimbus_lib as nl, nimbus_clean as nc, nimbus_train as nt
    from transformers import AutoTokenizer
    try:
        import peft
        peft_v = peft.__version__
    except ImportError:
        sys.exit("peft is not installed")
    sys.path.insert(0, a.pack)
    import score as scorer

    cuda = torch.cuda.is_available()
    if not cuda and not a.smoke:
        sys.exit("No CUDA GPU found. Use --smoke only for a CPU test.")
    gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
    env = {"hardware": gpu, "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if cuda else None,
           "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft_v, "python": platform.python_version(),
           "platform": platform.platform(), "note": "NON-T4 run: label as such in the report"}
    print("environment:", env)
    if cuda and "T4" not in gpu:
        print(f"NOTE: {gpu} is not a T4. Dev scores are comparable; memory/time are not.")

    out = os.path.abspath(a.out)
    for d in ("adapters", "preds", "logs"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    # ---- data: same cleaning as the notebook (Part A)
    nl.ROOT = os.path.join(a.pack, "data") + os.sep
    raw, dev = nl.load("train.jsonl"), nl.load("dev.jsonl")
    clean, log, stats = nc.clean_dataset(raw, dev)
    print(f"cleaning: {len(raw)} -> {len(clean)} rows  {dict(stats)}")
    assert stats["drop_unexplained"] == 0
    if not a.smoke:
        assert len(clean) == 1320, "unexpected cleaned size: is this the original train.jsonl?"
    raw_texts = {r["messages"][1]["content"] for r in raw}
    leak_ids = {r["id"] for r in dev if r["messages"][1]["content"] in raw_texts}
    train_rows = clean[: a.limit_train] if a.limit_train else clean
    dev_rows = dev[: a.limit_dev] if a.limit_dev else dev

    model_id = a.model or nt.MODEL_ID
    tok = AutoTokenizer.from_pretrained(model_id)

    def evaluate(texts):
        preds = [{"id": g["id"], "output": t} for g, t in zip(dev_rows, texts)]
        res = scorer.score(dev_rows, preds)["all"]
        keep = [g for g in dev_rows if g["id"] not in leak_ids]
        res_c = scorer.score(keep, preds)["all"] if keep else res
        return {"exact": res["exact_match"], "field_acc": res["mean_field_acc"], "json_valid": res["json_valid"],
                "exact_clean_dev": res_c["exact_match"], "fields": res["fields"]}, preds

    results_path = os.path.join(out, "results.jsonl")
    for name in [j.strip() for j in a.jobs.split(",") if j.strip()]:
        if name not in JOBS:
            print(f"skipping unknown job {name!r}; choose from {list(JOBS)}")
            continue
        cfg = {**DEFAULTS, **{k: v for k, v in JOBS[name].items() if k != "desc"}}
        print(f"\n=== {name}: {JOBS[name]['desc']}  cfg={cfg}")
        try:
            nt.free_memory()
            if cuda:
                assert torch.cuda.memory_allocated() < 0.5 * 2**30, "GPU memory not free at job start"
                torch.cuda.reset_peak_memory_stats()
            nt.seed_everything(cfg["seed"])
            base = tiny_base() if a.smoke else nt.load_base(model_id, quant4=cfg["quant4"])
            model = nt.add_lora(base, r=cfg["r"], dropout=cfg["dropout"], quant4=cfg["quant4"])
            trainable, total = nt.count_params(model)
            assert trainable <= 50e6, f"{trainable / 1e6:.1f}M trainable parameters exceeds the 50M cap"
            rows = reorder(train_rows) if cfg["reorder"] else train_rows
            exs = [nt.build_example(x, tok) for x in rows]
            print(f"  rows {len(exs)} | trainable {trainable / 1e6:.2f}M of {total / 1e6:.0f}M")
            tr = nt.train(model, tok, exs, epochs=1 if a.smoke else cfg["epochs"], lr=cfg["lr"], batch_size=16, seed=cfg["seed"])
            train_min, peak = tr["seconds"] / 60, nt.peak_gb()
            model.save_pretrained(os.path.join(out, "adapters", name))
            t0 = time.time()
            texts = nt.generate(model, tok, dev_rows, batch_size=16, max_new_tokens=20 if a.smoke else 200)
            ev, preds = evaluate(texts)
            texts2 = nt.generate(model, tok, dev_rows, batch_size=16, max_new_tokens=20 if a.smoke else 200, repetition_penalty=1.0)
            ev2, _ = evaluate(texts2)
            entry = {"name": name, "desc": JOBS[name]["desc"], "hardware": gpu, "cfg": cfg, "n_train": len(exs), "trainable_M": trainable / 1e6,
                     **ev, "exact_rep1.0": ev2["exact"], "field_acc_rep1.0": ev2["field_acc"], "peak_gb": peak, "train_min": train_min,
                     "eval_min": (time.time() - t0) / 60, "skipped_steps": tr["skipped_steps"],
                     "final_loss": float(np.mean([l for _, l in tr["loss_hist"][-10:]]))}
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            with open(os.path.join(out, "preds", f"{name}_dev.jsonl"), "w", encoding="utf-8") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            if a.test:                                          # test predictions: same call as the notebook (plain greedy, batch size 1)
                test_rows = [json.loads(l) for l in open(a.test, encoding="utf-8") if l.strip()]
                if a.limit_dev:
                    test_rows = test_rows[: a.limit_dev]
                t1 = time.time()
                test_texts = nt.generate(model, tok, test_rows, batch_size=1, max_new_tokens=20 if a.smoke else 200)
                n_valid = 0
                with open(os.path.join(out, "preds", f"{name}_test.jsonl"), "w", encoding="utf-8") as f:
                    for r, o in zip(test_rows, test_texts):
                        f.write(json.dumps({"id": r["id"], "output": o}, ensure_ascii=False) + "\n")
                        try:
                            n_valid += int(isinstance(json.loads(o), dict) and len(json.loads(o)) == 8)
                        except ValueError:
                            pass
                entry["test_rows"], entry["test_valid_8key"], entry["test_min"] = len(test_rows), n_valid, (time.time() - t1) / 60
                with open(results_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"name": name + "__test", "hardware": gpu, "test_rows": len(test_rows),
                                        "test_valid_8key": n_valid, "test_min": entry["test_min"]}) + "\n")
                print(f"  -> test predictions: {n_valid}/{len(test_rows)} valid 8-key JSON in {entry['test_min']:.1f} min")
            with open(os.path.join(out, "logs", f"{name}_loss.json"), "w") as f:
                json.dump(tr["loss_hist"], f)
            print(f"  -> exact {ev['exact']:.1%} | field {ev['field_acc']:.1%} | valid {ev['json_valid']:.1%} | clean-dev exact "
                  f"{ev['exact_clean_dev']:.1%} | rep_penalty=1.0 exact {ev2['exact']:.1%} | peak {peak:.1f} GB | train {train_min:.1f} min")
            del model, base
        except Exception as e:                                  # one failing job must not waste the others
            import traceback
            traceback.print_exc()
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"name": name, "hardware": gpu, "error": repr(e)}) + "\n")
        finally:
            nt.free_memory()

    # ---- summary + zip (no company data inside)
    rows = [json.loads(l) for l in open(results_path, encoding="utf-8")] if os.path.exists(results_path) else []
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "hardware", "exact", "field_acc", "json_valid", "exact_clean_dev", "exact_rep1.0", "peak_gb", "train_min", "error"])
        for r in rows:
            w.writerow([r.get(k, "") for k in ("name", "hardware", "exact", "field_acc", "json_valid", "exact_clean_dev", "exact_rep1.0", "peak_gb", "train_min", "error")])
    with open(os.path.join(out, "environment.json"), "w") as f:
        json.dump(env, f, indent=1)
    zpath = os.path.join(out, "college_results.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, _, files in os.walk(out):
            for fn in files:
                full = os.path.join(folder, fn)
                if full != zpath:
                    z.write(full, os.path.relpath(full, out))
    print(f"\nDONE. Copy this file back to the laptop: {zpath}  ({os.path.getsize(zpath) / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
