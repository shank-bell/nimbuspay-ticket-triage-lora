# NimbusPay ticket triage: LoRA fine-tune of Qwen2.5-1.5B-Instruct

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Loaded in 4-bit:** **No.** The final adapter was trained and is evaluated with the base model loaded in **fp16** (LoRA r=16 on all linear layers, 18.46M trainable parameters, 5 epochs). A 4-bit QLoRA variant was an experiment only (88.0% vs 89.0% at 3 epochs).
- **Best result (one line):** **94.0% exact match / 98.9% mean field accuracy** on the 200 dev tickets (clean data, 5 epochs), from the fresh top-to-bottom run of the final notebook on a Colab T4 (my earlier experiment run of the same settings: 93.0 / 98.6). Base model with the fixed prompt: 0%. Test: 400 predictions, all valid 8-key JSON.
- **Inference as specified:** model's own chat template on `[system, user]`, plain greedy `model.generate`, `max_new_tokens=200`, only `.strip()` on the output, no rules in the prompt, no constrained decoding, no post-fixing.

## Files
| File | What |
|---|---|
| `notebook.ipynb` | Colab notebook with outputs; runs top to bottom on a fresh T4 (Part A audit and cleaning, B baselines, C pipeline evidence, D experiments from the embedded log (set `RUN_EXPERIMENTS = True` to re-run), E error analysis, final model and test predictions) |
| `report.md` | audit findings, experiment table, error analysis, next steps, AI tools used |
| `cleaning_log.csv` | `id,action,reason` for all 300 dropped/fixed training rows |
| `adapter/` | final PEFT adapter (`adapter_config.json` + `adapter_model.safetensors`) |
| `predictions_test.jsonl` | `{"id", "output"}` for the 400 test tickets (raw model output) |
| `audit/`, `src/`, `tools/`, `tests/`, `college/` | audit and cleaning code (`nimbus_lib.py`, `nimbus_clean.py`), training loop (`nimbus_train.py`), notebook builder, CPU tests, runner used for non-T4 repeats |

The company's training data and the base model are not in this repository.

## Reproduce
Upload the candidate pack to `MyDrive/nimbus/candidate_pack`, open `notebook.ipynb` in Colab, select a T4 and *Run all*. Colab's `torchao 0.10` must be removed for the current `peft` (the first cell does it).

## How the predictions were made, and what could change them
`predictions_test.jsonl` is the raw output of the final adapter on a fresh Colab T4 (torch 2.14.1+cu130, transformers 5.22.2, peft 0.21.2): base model in fp16 + `PeftModel.from_pretrained("adapter")`, the model's own chat template on `[system, user]` with `add_generation_prompt=True`, `model.generate(do_sample=False, max_new_tokens=200)` at batch size 1, only `.strip()` afterwards. The model's default `generation_config` stays in force, so `repetition_penalty=1.1` (Qwen's default) is active. Stability checks on the 400 test tickets: batch size 16 gives identical text on 400/400; `repetition_penalty=1.0` gives identical text on 388/400 (12 differ). A different GPU, dtype or library version could still change a few outputs; that was not measured.

## Known limitations
- The final model is one training seed, and dev has 200 tickets (one ticket = 0.5 points), so 94.0 vs 93.0 is 2 tickets. A seed-to-seed spread of 2.5 points was measured on the non-T4 repeats (3-epoch runs).
- Five test predictions break SCHEMA.md rules (listed in `report.md`, Section 6). They are left exactly as generated, because post-fixing is not allowed.
