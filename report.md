# NimbusPay ticket triage: fine-tuning Qwen2.5-1.5B-Instruct (report)

Every number below was copied from the notebook outputs (final fresh-T4 run, 2026-11-12) or from the experiment log; non-T4 numbers are labelled.

## 1. Summary
Base model `Qwen/Qwen2.5-1.5B-Instruct`, LoRA r=16 on all linear layers (18.46M trainable parameters, cap 50M), own PyTorch + PEFT loop, fp16 on a free Colab T4. An audit turned 1,445 training rows into 1,320 clean rows (worth about +10 points). On the 200 dev tickets (strict scoring) the final model (clean data, 5 epochs) reaches **94.0% exact match / 98.9% mean field accuracy** (93.5% exact on the 170 clean dev tickets) in the fresh top-to-bottom run of the final notebook; the earlier experiment run of the same settings (D3b) gave 93.0 / 98.6. All 400 test predictions are valid 8-key JSON. Base model with the fixed prompt: 0%; with all rules in the prompt: 0% strict exact match.

## 2. Data audit (Part A)
Method: structural profile; template mining (tickets come from a fixed phrase inventory); a rule checker that re-derives every deterministic field from the ticket text using only SCHEMA.md; a 5-fold cross-validated classifier for category; duplicate and dev-leakage checks. Code `audit/`, log `cleaning_log.csv` (300 rows: 125 dropped, 175 fixed, 1,320 kept).

| Problem | Rows | Action |
|---|---|---|
| Label JSON truncated | 25 | drop |
| Legacy v1 schema (`prio` low/medium/high, `dd/mm/yyyy`) | 70 | fix (mapping equals the spec-derived priority on 70/70) |
| Junk tickets (`test`, `asdf asdf`, empty) with filled labels | 10 | drop |
| Exact duplicates | 60 | drop second copy |
| Train tickets identical to dev tickets (leakage; none overlap test) | 30 | drop |
| `amount_paise` stored in rupees (incl. `k`, `lakh`) | 25 | fix |
| `txn_id` copied raw (`np 81633 47892`) | 20 | normalise |
| `priority` contradicts the spec | 60 | fix; `needs_human` equals the derived value in 60/60 |

Checked and clean: category (CV agreement 97.9%; all 29 disagreements read by hand, none wrong), channel, language, dates, `needs_human`, formatting. The deriving code reproduces the dev gold labels on all 7 deterministic fields at 100% (dev labels used for measurement only; dev inputs, never labels, found the 30 leaked rows).

## 3. Pipeline (Part C)
- Stack: plain PyTorch + Hugging Face PEFT with my own training loop (no Trainer/TRL), because (a) it stays stable while Colab upgrades library versions, (b) the loss is computed only at assistant positions, so the 152k-vocab logits are never built for prompt tokens (a 2,200-token quoted-thread ticket would need about 1.3 GB of fp32 logits), and (c) every line is explainable in the live defense. Qwen's template has a system role, so the fixed system prompt is used as given.
- The model's own chat template on `[system, user]` + generation prompt (what `generate` receives); target = JSON + `<|im_end|>`. Loss only on assistant tokens (my loss equals the Hugging Face loss to 6 decimals in a CPU test).
- Sequence length from evidence: 97% of examples are at most 263 tokens, 38 tickets with quoted `>` threads are 1,000-2,214 tokens and the label is at the end, so truncation would delete the target: `MAX_LEN = 2560`, nothing truncated; 4,608-token micro-batches, gradient checkpointing, fp16 autocast with fp32 LoRA weights.
- Budget: all linear layers, r=8/16/32/64 = 9.2/18.5/36.9/73.9M parameters, so r=64 would break the 50M cap.
- AdamW, lr 2e-4, cosine, 5% warmup, 16 sequences per step; greedy decoding with the model's own generation_config (`repetition_penalty=1.1`).

## 4. Baselines (Part B, dev, strict)
| | exact | field acc | valid JSON |
|---|---|---|---|
| B1 base model, fixed prompt | 0.0% | 0.0% | 0% (own keys, JSON in code fences) |
| B2 base model, SCHEMA.md rules in prompt | 0.0% | 19.4% | 34% (lenient parse: field acc 59.9%, exact 0.5%) |

## 5. Controlled experiments (Part D)
Colab T4, seed 0, same 200 dev tickets (one ticket = 0.5 points; a 95% interval near 90% is about +-4 points). "rep 1.0" = scored with `repetition_penalty=1.0` instead of the default 1.1.

| Run (one change vs R0) | exact | field | clean-dev | rep 1.0 | peak GPU | train min |
|---|---|---|---|---|---|---|
| R0 clean data, 16-bit LoRA, 3 epochs | 89.0 | 97.9 | 88.8 | 90.5 | 5.3 GB* | 14.3 |
| D1 raw uncleaned data | 79.0 | 96.5 | 77.6 | 87.0 | 5.3 GB | 15.4 |
| D2 4-bit QLoRA | 88.0 | 97.7 | 88.2 | 90.5 | 4.7 GB | 17.7 |
| D3a 1 epoch | 42.0 | 90.1 | 42.4 | 48.0 | 5.3 GB | 4.8 |
| D3b 5 epochs | 93.0 | 98.6 | 92.4 | 93.0 | 5.4 GB | 23.8 |
| D4 priority, needs_human emitted last | 87.0 | 97.5 | 87.6 | 89.0 | 5.3 GB | 14.3 |
| *Final notebook, fresh T4:* R0 re-run | 89.0 | 97.8 | 88.8 | 90.0 | 5.33 GB | 14.5 |
| *Final notebook, fresh T4:* **FINAL (D3b settings)** | **94.0** | **98.9** | **93.5** | 94.0 | 5.4 GB | 24.1 |

\*R0's first log said 8.2 GB, inflated by about 3 GB left over from a failed attempt in the same session; the live re-run measured 5.33 GB. The same seed on a fresh runtime is not bit-identical (R0 field 97.8 vs 97.9, rep 1.0 90.0 vs 90.5; D3b settings 94.0 vs 93.0, i.e. 2 tickets), so run-to-run GPU nondeterminism is about 1 point.

Non-T4 repeat (college PC, RTX PRO 4000 Blackwell 24 GB, torch 2.11 instead of 2.14; **not comparable for memory/time**): R0 seed 0 **89.0** (same loss curve as the T4), R0 seed 1 **91.5**, D4 87.5, D3b 92.0.

Findings.
1. **Cleaning: +10 points** (D1), well above noise; the raw-trained model is also fragile to the decoding config (+8 points without the repetition penalty, vs +1.5 for the clean model).
2. **QLoRA** costs about 1 point (within noise), saves 0.6 GB and is 24% slower; the saving is small, probably because activations and the fp32 upcast dominate (not measured separately).
3. **Epochs.** My hypothesis (one epoch is enough, more over-fits) was wrong: after 1 epoch the loss (0.024) looks small but exact match needs every digit right; it reaches ~0 only after 3-5 epochs (final loss 0.0240 / 0.0013 / 0.0002). 5 epochs also removes the repetition-penalty sensitivity (94.0 with and without). The final model uses 5 epochs by a rule fixed in advance (5 epochs only if clean-dev exact beats R0's by 2 points: 92.4 vs 88.8). **Caution:** seeds 0 and 1 differ by 2.5 points, so the gain (+4.0 and +5.0 on the T4; mean of three 5-epoch runs 93.0 vs four 3-epoch runs 89.6, though repeats of one seed are not independent) is suggestive, not conclusive.
4. **D4 (my idea) is a negative result** (87.0 T4, 87.5 college): priority errors are not caused by deciding before writing the amount.

## 6. Error analysis (Part E, final model, 200 dev tickets)
12 of 200 tickets have a wrong field (6 with one, 6 with two): priority 8, category 4, needs_human 4, amount_paise 1, txn_date 1, none in txn_id, channel, language. Per-field accuracy (derived from these counts): priority 96.0, category 98.0, needs_human 98.0, amount 99.5, date 99.5, other three 100 (mean 98.9). Paired with the live R0 (22 wrong): only R0 wrong 11, only FINAL wrong 1, both wrong 11, both right 177. The 12 failures (ticket text read from the dev file; "why" is my reading, not proof):

| id | wrong | gold -> predicted | why (likely) |
|---|---|---|---|
| dv-00017 | amount | 18082725 -> 18082525 | "Rs 1,80,827.25" (lakh grouping + decimals): one digit mis-copied; exact match needs every digit |
| dv-00019 | category | refund -> `product_fault` | value not in the schema; text says the product was faulty, no refund word (scorer checks types only, so a wrong value, not invalid JSON) |
| dv-00037 | priority, needs_human | P1 -> P2, true -> false | "284669 rupees" without separators: Rs 50,000 rule missed |
| dv-00046 | category, priority | fraud -> refund, P1 -> P3 | "a payment on my statement is wrong", no "I did not make this"; fraud wins, priority follows |
| dv-00056 | priority, needs_human | P2 -> P1, false -> true | Rs 44,206.75 is under the Rs 50,000 line but looks large |
| dv-00090 | txn_date | 2026-07-10 -> 2026-07-11 | Hinglish "parso" (day before yesterday) read as "kal" |
| dv-00127 | priority | P3 -> P2 | Rs 4,836 just under the Rs 5,000 line; distractor "account balance Rs 111" |
| dv-00129 | priority, needs_human | P2 -> P1, false -> true | Rs 49,657 just under the Rs 50,000 line |
| dv-00148 | priority | P2 -> P1 | Rs 42,062 refund, "second complaint", long quoted thread (rare: 38 in train) |
| dv-00157 | category, priority | fraud -> account_access, P1 -> P2 | "OTP maang ke paise nikal liye": the OTP word pulled it to account_access; money taken without consent is fraud |
| dv-00177 | category | payment_failed -> fraud | distractor "a red flag on my top up"; the top-up itself failed (P1 either way, Rs 2.9 lakh) |
| dv-00180 | priority, needs_human | P3 -> P1, false -> true | amount copied right (313000 paise) but priority driven by the distractor "account balance Rs. 241670" |

Patterns. (1) 8 of 12 tickets involve priority; 6 of those 8 are amounts near the Rs 5,000 / 50,000 thresholds or a distractor number, the other 2 follow a wrong category. (2) All 4 needs_human errors follow a wrong priority. (3) The 4 category errors are fraud / refund / payment_failed boundary cases. (4) Only 2 of 200 tickets have an error in a copied field (one digit, one relative date): surface forms are learned well; what is left is threshold logic and Hinglish phrasing.

Test predictions (400 rows, checked against SCHEMA.md without labels; none exist). All 400 are valid 8-key JSON and the ids match the test file in order. Five rows break a schema rule that is visible without labels: ts-00001 `txn_date` `2026-07-00` (a non-existent day; "Received 2026-07-06, 6 din pehle" should give 2026-06-30 by the relative-date rule); ts-00209 `txn_id` `NP12077555550` has 11 digits (the ticket writes `np 12077 55550`, 10 digits, so one digit was duplicated while normalising); ts-00367 `channel` `apple_pay` (not in the enum; the ticket says "American Express in Apple Pay"); ts-00008 and ts-00377 have category `other` with non-null transaction fields (the spec says they are null for `other`; the tickets are about a missing scratch-card reward and a pending transaction, so `other` itself is doubtful, though I have no label to prove it). The submission is the raw model output, so none of these were repaired.

## 7. What I would do with one more week
(a) Augment from the spec's own variation list (ids, dates, channels, legal threats), because test inputs contain many sentence templates that never occur in train (38% vs 24% for dev, measured on inputs only); (b) loss weighting or a curriculum on the priority thresholds (Rs 5,000 / 50,000), where most errors are; (c) 3-5 seeds per variant to separate effects from noise; (d) a held-out "hard slice" of paraphrases to track the shift; (e) training tickets with relative dates that cross month and year boundaries ("6 din pehle" on the 6th) and with ids written with spaces, plus enum-valid channel examples (the 5 test rows above); (f) a rule-based date and amount verifier used at evaluation time only, to measure how many errors it would catch (not part of the submission).

## 8. AI tools and honesty notes
An AI coding assistant (an LLM agent) wrote the audit scripts, notebook, training module and tests, and drove the Colab runs through browser automation; I reviewed them and must be able to defend every line and number. A second session of the same assistant on a college PC ran the non-T4 repeats with a prepared script. Notes: (1) I inspected test *inputs* (never labels; none exist) to measure the paraphrase shift; no test row was used for training or tuning. (2) Dev inputs found the 30 leaked rows; dev labels only validated the checker, never decided a fix. (3) Most T4 runs use one seed; the seed-noise measurement is non-T4. (4) R0's first memory figure is wrong (Section 5). (5) The college `results.jsonl` keeps one stale error row from a failed first attempt (missing `jinja2`). (6) The submitted predictions use the default generation config, plain `generate`, batch size 1 (400 rows, 400 valid 8-key JSON, 33.4 min on the T4); they equal batch-size-16 output on 400/400 tickets and `repetition_penalty=1.0` output on 388/400 (12 differ), measured in the last notebook cell. (7) The final model is one seed; on 200 dev tickets 94.0 vs 93.0 is 2 tickets, not a real gain.
