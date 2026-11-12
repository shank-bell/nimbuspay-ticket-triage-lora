# Part A – Data audit findings (train.jsonl, 1,445 rows → 1,320 clean rows)

Reproduce: `cd audit` then `uv run --python 3.11 python -X utf8 <script>` for `01_profile.py`, `02_templates.py`, `03_rulecheck.py`,
`04_category_cv.py` (needs `--with scikit-learn --with numpy`), `05_clean.py`, `06_validate.py`. `lib.py` holds the SCHEMA.md rules.

## Method
1. **Structural profile** (01): JSON validity, key sets, types, enums, header format, system prompt, ids.
2. **Template mining** (02): replace digits by `#`, count sentences. The tickets are generated from a fixed phrase inventory
   (amount/date/id/channel/contact/legal/distractor phrases), which made exact extractors possible.
3. **Rule checker** (03, `lib.py`): re-derive every deterministic field from the ticket text using only SCHEMA.md
   (amount incl. k/lakh/decimals and distractors; dates incl. relative, no-year; NP ids; channel; Hinglish; legal/contact → priority, needs_human).
4. **Truth-first comparison**: judge each field against the spec-derived truth, not against the other (possibly corrupted) label fields.
   This localises which column is corrupt (e.g. `needs_human` always matches the derived priority, so `priority` is the bad column).
5. **Category** (04): 5-fold CV TF-IDF logistic regression on train only (97.9% agreement) → manually read all 29 disagreements;
   plus a fraud-precedence cue check. Nothing was mislabelled.
6. **Validation** (06): checker re-run on cleaned file → 0 violations. `derive()` reproduces dev gold at 100% on all 7 deterministic fields
   (dev labels used for *measurement only*, never to decide a fix).

## Problems found
| # | Problem | Rows | How found | Decision |
|---|---|---|---|---|
| 1 | Label JSON truncated mid-string (`json.loads` fails) | 25 | 01: parse each assistant message | **drop** (cannot trust partial label) |
| 2 | Legacy v1 schema: key `prio` = low/medium/high, dates `dd/mm/yyyy` | 70 | 01: key-set Counter showed 1350 vs 70 | **fix**: rename, low/medium/high→P3/P2/P1, date→ISO. Mapping agrees with spec-derived priority on 70/70 and every other field passes |
| 3 | Junk tickets: body `test`, `test test 123`, `asdf asdf` or empty, yet the label has amount/date/id | 10 | 02 template mining + 03 "label amount/id/date not in text" | **drop** (label unsupported by text) |
| 4 | Exact duplicate rows (same text and label) | 60 | 05: exact full-text match | **drop** later copy (keep first) |
| 5 | **Leakage**: train ticket text identical to a dev ticket (0 overlap with test) | 30 | 05: exact match of train user text against dev *inputs* | **drop** (never train on dev) |
| 6 | `amount_paise` holds rupees (₹674 → 674; `1.1k` → 1100; `2.2 lakh` → 220000; `Rs 594.50` → 594) | 25 | 03: label amount not among text amounts; label == floor(text_paise/100) in 25/25 | **fix** to text-derived paise |
| 7 | `txn_id` copied raw, not normalised (`NP-9718716109`, `np9891267518`, `NP 9255627510`, `np 81633 47892`) | 20 | 03: regex `NP\d{10}` format check | **fix**: uppercase, digits only |
| 8 | `priority` contradicts the spec (label P1/truth P3: 31, P3/P1: 15, P3/P2: 14) | 60 | 03 truth-first: recompute from category + text amount + legal phrase | **fix**: recompute. Evidence that it is noise: spread across id range, categories and sources; `needs_human` equals the spec-derived value in 60/60, and in the 43 rows where the corrupted priority would imply a *different* `needs_human` (the other 17 are P2/P3 flips or contact-phrase rows where both priorities imply the same value) it matches the derived priority 43/43 and the corrupted one 0/43 |

Totals: 125 dropped (25+10+60+30), 175 fixed (70+25+20+60), 1,320 kept. Every row appears once in `cleaning_log.csv` (id, action, reason).
Zero rows were dropped as "unexplained": every mismatch matched a known corruption signature.

## Checked and found clean
- `category`: CV 97.9%; 29 disagreements read by hand, none wrong (genuinely ambiguous Banking77-style phrasing). No non-fraud row contains fraud language (spec: fraud wins).
- `needs_human`: 0 errors vs spec-derived value. `language`, `txn_date` (relative/absolute/no-year), `channel` (incl. "my credit card was declined" → card, "scratch card" → null): 0 errors apart from the junk rows.
- Formatting: all labels are canonical `json.dumps` with the fixed key order; weekdays in headers are correct; no future txn dates; no non-positive amounts; one system prompt.

## Observations that affect later parts
- **Long inputs**: 38 clean tickets (2.9%) contain quoted `>` email threads of up to 8.8k chars (≈2k tokens); the other 97% are under ~330 chars. Truncating the end of the sequence would cut the label, so sequence length must be chosen from token counts (Part C).
- **Order of the long threads**: the quoted text is boilerplate support replies; dates and ticket numbers inside it must be ignored (spec rule).
- **Train vs dev** distributions are close after cleaning (priority P1 19.5/21, P2 24.5/26, P3 56/53 %; Hinglish 28.5/24.5 %).
- **Test shift** (inputs only, no labels): 38% of test sentence templates never occur in train versus 24% for dev, and 95% of test tickets have at least one unseen sentence versus 73% for dev.
  The unseen material is paraphrased wording of the same slots (see the honesty note in the report).
- **Honesty note**: I looked at test *inputs* to measure this shift. I did not use test labels (none exist), did not train on test rows, and the augmentation (Part D) must be written from SCHEMA.md-listed variations, not copied from test sentences.
