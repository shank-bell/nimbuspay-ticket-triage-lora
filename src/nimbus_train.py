"""Training / generation helpers for the NimbusPay triage fine-tune (Qwen2.5-1.5B-Instruct + LoRA / QLoRA).

Why a hand-written loop instead of Trainer/TRL
  * API-stable across library versions (Colab upgrades them under us),
  * the loss is computed ONLY on the assistant tokens and ONLY at those positions, so the 152k-vocab logits are never
    materialised for prompt tokens (a 2,200-token quoted-thread ticket would otherwise need ~1.3 GB of fp32 logits),
  * every line is explainable in the live defense.
"""
import gc, json, math, random, re, time
import torch
import torch.nn.functional as F

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------- data -> tensors
def build_example(row, tok):
    """Chat-template the system+user messages, append the assistant JSON + <|im_end|>; loss only on the completion.

    The prompt part is exactly what generate() receives at test time (add_generation_prompt=True), so training and
    inference see identical prefixes. Prompt and completion are tokenised separately; a check in the notebook shows
    this equals joint tokenisation on every row.
    """
    prompt = tok.apply_chat_template(row["messages"][:2], tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(row["messages"][2]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    return {"id": row["id"], "input_ids": p + c, "labels": [-100] * len(p) + c, "n_prompt": len(p), "n_target": len(c)}


def collate(exs, pad_id, device):
    L = max(len(e["input_ids"]) for e in exs)
    ids = torch.full((len(exs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(exs), L), dtype=torch.long)
    lab = torch.full((len(exs), L), -100, dtype=torch.long)
    for i, e in enumerate(exs):                      # right padding for training
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        att[i, :n] = 1
        lab[i, :n] = torch.tensor(e["labels"])
    return {"input_ids": ids.to(device), "attention_mask": att.to(device), "labels": lab.to(device)}


def micro_batches(exs, budget):
    """Split one optimiser batch into micro-batches of similar length with <= `budget` padded tokens each."""
    exs = sorted(exs, key=lambda e: len(e["input_ids"]))
    out, cur = [], []
    for e in exs:
        if cur and (len(cur) + 1) * len(e["input_ids"]) > budget:
            out.append(cur)
            cur = []
        cur.append(e)
    if cur:
        out.append(cur)
    return out


def causal_lm(model):
    """The underlying *ForCausalLM: unwraps a PeftModel (LoRA layers live inside it); a plain HF model is returned as is."""
    try:
        from peft import PeftModel
    except ImportError:
        return model
    return model.base_model.model if isinstance(model, PeftModel) else model


def loss_sum(model, batch):
    """Sum of token cross-entropy over the assistant tokens (labels != -100) and how many tokens that is."""
    inner = causal_lm(model)
    h = inner.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    tgt = batch["labels"][:, 1:]                                                  # position t predicts token t+1
    keep = tgt != -100
    logits = inner.lm_head(h[:, :-1][keep]).float()                               # only the completion positions
    return F.cross_entropy(logits, tgt[keep], reduction="sum"), int(keep.sum())


# --------------------------------------------------------------------------- model
def _dtype_kw(transformers, dtype):
    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype" if ver >= (4, 56) else "torch_dtype": dtype}


def load_base(model_id=MODEL_ID, quant4=False):
    import transformers
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kw = _dtype_kw(transformers, torch.float16)
    kw.update(attn_implementation="sdpa", device_map={"": 0})
    if quant4:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(model_id, **kw)


def add_lora(base, r=16, alpha=None, dropout=0.05, quant4=False, targets=LORA_TARGETS):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    gc_kw = {"use_reentrant": False}
    if quant4:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=gc_kw)
    else:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kw)
        base.enable_input_require_grads()
    base.config.use_cache = False
    cfg = LoraConfig(r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout, target_modules=targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    for p in model.parameters():                       # fp32 adapters + fp16 autocast = stable mixed precision on a T4
        if p.requires_grad:
            p.data = p.data.float()
    return model


def count_params(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return tr, tot


def prepare_for_generation(model):
    inner = causal_lm(model)
    inner.config.use_cache = True
    model.eval()
    return model


# --------------------------------------------------------------------------- training
def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train(model, tok, examples, *, epochs=3, lr=2e-4, batch_size=16, warmup_ratio=0.05, micro_budget=4608,
          weight_decay=0.0, max_grad_norm=1.0, seed=0, amp=True, log_every=10, verbose=True):
    """AdamW + cosine schedule with warmup. One optimiser step = `batch_size` sequences, token-mean loss."""
    from transformers import get_cosine_schedule_with_warmup
    dev = next(model.parameters()).device
    use_amp = bool(amp and dev.type == "cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total = steps_per_epoch * epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(warmup_ratio * total)), total)
    scaler = _grad_scaler(use_amp)
    rng = random.Random(seed)
    model.train()
    hist, skipped, step, t0 = [], 0, 0, time.time()
    for ep in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        run = []
        for b in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[b:b + batch_size]]
            n_tgt = sum(e["n_target"] for e in batch)
            tot_loss = 0.0
            for mb in micro_batches(batch, micro_budget):
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                    ls, _ = loss_sum(model, collate(mb, tok.pad_token_id, dev))
                scaler.scale(ls / n_tgt).backward()           # token-mean over the whole optimiser batch
                tot_loss += ls.item()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            before = scaler.get_scale() if use_amp else None
            scaler.step(opt)
            scaler.update()
            if use_amp and scaler.get_scale() < before:
                skipped += 1                                  # fp16 overflow -> step skipped by the scaler
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            loss = tot_loss / n_tgt
            run.append(loss)
            if not math.isfinite(loss) and verbose:
                print(f"  !! non-finite loss at step {step}")
            if verbose and (step % log_every == 0 or step == total):
                print(f"  ep {ep + 1}/{epochs} step {step:4d}/{total} loss {sum(run[-log_every:]) / len(run[-log_every:]):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} |g| {float(gn):.2f}  {(time.time() - t0) / 60:.1f} min")
            hist.append((step, loss))
    return {"loss_hist": hist, "seconds": time.time() - t0, "steps": step, "skipped_steps": skipped}


# --------------------------------------------------------------------------- generation
@torch.no_grad()
def generate(model, tok, rows, *, batch_size=16, max_new_tokens=200, system_override=None, **gen_kwargs):
    """Greedy decoding exactly as in the assignment: chat template on [system, user], do_sample=False, 200 new tokens.

    Nothing else is changed: the model's own generation_config (incl. its repetition_penalty) stays in force unless the
    caller passes an override, which is how the sensitivity check in the notebook works.
    Returns the raw decoded text with only .strip() applied.
    """
    prepare_for_generation(model)
    dev = next(model.parameters()).device
    old_side = tok.padding_side
    tok.padding_side = "left"
    prompts = []
    for r in rows:
        msgs = r["messages"][:2]
        if system_override is not None:
            msgs = [{"role": "system", "content": system_override}, msgs[1]]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    order = sorted(range(len(rows)), key=lambda i: -len(prompts[i]))      # similar lengths together -> little padding
    outs = [None] * len(rows)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **gen_kwargs)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            outs[i] = t.strip()
    tok.padding_side = old_side
    return outs


def lenient_json(text):
    """Diagnostic only (never used for the official score): strip code fences / prose around the first {...}."""
    if text is None:
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", t, re.S)
    return m.group(0) if m else t


# --------------------------------------------------------------------------- bookkeeping
def peak_gb():
    return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else float("nan")


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
