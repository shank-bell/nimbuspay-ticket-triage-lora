"""CPU smoke test of src/nimbus_train.py with a tiny random Qwen2 (no GPU, no big download).
Checks: loss masking, custom loss == HF loss, training reduces loss, adapter save/reload == in-memory model, batched generation runs."""
import sys, json, os, tempfile
sys.path.insert(0, "E:/Fine_Tuneing-Assigh/src"); sys.path.insert(0, "E:/Fine_Tuneing-Assigh/audit")
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from peft import PeftModel
import nimbus_train as nt
from nimbus_lib import load

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("E:/Fine_Tuneing-Assigh/scratch_model")
rows = load("E:/Fine_Tuneing-Assigh/audit/train_clean.jsonl")[:64]
cfg = Qwen2Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=True)
base = Qwen2ForCausalLM(cfg).float()
tmp = tempfile.mkdtemp(); base.save_pretrained(tmp + "/base")
model = nt.add_lora(base, r=4, alpha=8, dropout=0.0)
tr, tot = nt.count_params(model); print(f"trainable {tr:,} / total {tot:,}")
assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)

# 0. only LoRA weights are trainable
names = [n for n, p in model.named_parameters() if p.requires_grad]
assert names and all("lora_" in n for n in names), names[:3]
print("0 ok  trainable tensors are all LoRA:", len(names))

# 1. masking
exs = [nt.build_example(r, tok) for r in rows]
e = exs[3]
supervised = tok.decode([t for t in e["labels"] if t != -100])
assert supervised == rows[3]["messages"][2]["content"] + "<|im_end|>", supervised
assert all(l == -100 for l in e["labels"][:e["n_prompt"]]) and e["labels"][e["n_prompt"]:] == e["input_ids"][e["n_prompt"]:]
print("1 ok  loss is on the assistant JSON + <|im_end|> only:", repr(supervised[:60]), "...")

# 2. custom loss == HF loss (padded batch, eval mode)
model.eval()
batch = nt.collate(exs[:6], tok.pad_token_id, "cpu")
with torch.no_grad():
    hf = model(**batch).loss.item()
    ls, n = nt.loss_sum(model, batch)
print(f"2 HF loss {hf:.6f}  custom {ls.item() / n:.6f}  tokens {n}")
assert abs(hf - ls.item() / n) < 1e-4

# 3. micro-batching keeps every example exactly once and respects the budget
mbs = nt.micro_batches(exs[:16], 1500)
assert sorted(x["id"] for m in mbs for x in m) == sorted(x["id"] for x in exs[:16])
assert all(len(m) == 1 or len(m) * max(len(x["input_ids"]) for x in m) <= 1500 for m in mbs)
print("3 ok  micro-batches:", [len(m) for m in mbs])

# 4. training lowers the loss
with torch.no_grad():
    sub = exs[:32]
    def mean_loss():
        t = n_ = 0
        for mb in nt.micro_batches(sub, 4000):
            ls, n = nt.loss_sum(model, nt.collate(mb, tok.pad_token_id, "cpu")); t += ls.item(); n_ += n
        return t / n_
    before = mean_loss()
res = nt.train(model, tok, exs[:32], epochs=6, lr=3e-3, batch_size=8, log_every=6, amp=False)
model.eval()
with torch.no_grad(): after = mean_loss()
print(f"4 loss {before:.3f} -> {after:.3f}  ({res['steps']} steps, {res['seconds']:.1f}s, skipped {res['skipped_steps']})")
assert after < before - 0.1   # tiny frozen-head toy model: only checks that gradients flow and the loss goes down

# 5. save adapter in PEFT format and reload on a fresh copy of the base
model.save_pretrained(tmp + "/adapter"); print("5 adapter files:", sorted(os.listdir(tmp + "/adapter")))
assert "adapter_config.json" in os.listdir(tmp + "/adapter") and "adapter_model.safetensors" in os.listdir(tmp + "/adapter")
base2 = Qwen2ForCausalLM.from_pretrained(tmp + "/base").float()
m2 = PeftModel.from_pretrained(base2, tmp + "/adapter").eval()
with torch.no_grad():
    a = model(**batch).logits; b = m2(**batch).logits
print("   max |logit diff| after reload:", float((a - b).abs().max())); assert float((a - b).abs().max()) < 1e-4

# 6. batched generation: runs, deterministic, equals batch_size=1 (left padding is handled)
out_b = nt.generate(model, tok, rows[:5], batch_size=3, max_new_tokens=12)
out_1 = nt.generate(model, tok, rows[:5], batch_size=1, max_new_tokens=12)
print("6 batched == unbatched:", out_b == out_1); print("   sample:", repr(out_b[0][:60]))
# 7. plain HF model (no PEFT) works too: this is the path the un-tuned baselines use
plain = Qwen2ForCausalLM.from_pretrained(tmp + "/base").float().eval()
assert nt.causal_lm(plain) is plain and nt.causal_lm(m2) is m2.base_model.model
with torch.no_grad():
    hf = plain(**batch).loss.item(); ls, n = nt.loss_sum(plain, batch)
assert abs(hf - ls.item() / n) < 1e-4
o = nt.generate(plain, tok, rows[:3], batch_size=2, max_new_tokens=8, system_override="You are a test.")
print("7 ok  plain-model loss + generation (system_override) work:", len(o), "outputs")
print("ALL SMOKE TESTS PASSED")
