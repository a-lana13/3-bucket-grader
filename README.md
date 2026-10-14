# Harm Severity Grader

Grades free-text patient safety incident reports on the 10-grade PSRS harm
scale (A through I) so reviewers can triage the highest-risk reports first
instead of reading every one. One model produces three outputs:

- **Letter grade** (A, B1, B2, C, D, E, F, G, H, I)
- **3-bucket label**: `none` / `some` / `serious`
- **Triage score**: a calibrated probability that the patient was harmed,
  plus a flag for priority review

This repo covers the severity/triage requirement. The teammate's `11-buckets`
repo covers event-type and recurring-theme classification. The two answer
different sponsor requirements and stay separate.

See [PIPELINE.md](PIPELINE.md) for diagrams and full per-experiment detail.

## Data

Synthetic MIDAS-style dataset: 80,000 reports, 23 columns. 67,003 are usable
after dropping rows with no harm score or no narrative text.

| Bucket | Grades | Meaning |
|---|---|---|
| `none` | A, B1, B2, C, D | Unsafe condition, near miss, or no harm |
| `some` | E, F | Temporary harm, treated (incl. added hospitalization) |
| `serious` | G, H, I | Permanent harm, near death, or death |

**Split by date**, decoded from the `Event No.` prefix, so the model is always
tested on reports from a later month than it trained on:

| Bucket | Train (Jan–Aug) | Val (Sep) | Test (Oct) |
|---|---|---|---|
| `none` | 42,523 (84.9%) | 8,245 (95.9%) | 7,909 (95.2%) |
| `some` | 7,209 (14.4%) | 339 (3.9%) | 383 (4.6%) |
| `serious` | 359 (0.7%) | 16 (0.2%) | 20 (0.2%) |

Training over-samples harm cases on purpose; validation and test reflect the
natural rate.

## Model inputs (leakage-safe)

The model only reads what's available when a report is submitted:

- `event_comments`, the reporter's narrative
- A short labeled prefix of intake fields: unit (`Location Name`), service
  (`Encounter Service`), age band, and prescribed / administered / suspect
  medication and dose

These fields are **never** used, because they're filled in after a safety
officer reviews the report and would leak the answer:
`manager_comments`, `unit_actions_taken`, `shareable_lessons`,
`HPI Designation`, `Analyst-Report Type`, and Level of Investigation
(`Level of Invet`). These are the `EXCLUDED_COLUMNS` in `data_pipeline.py`,
which asserts on every load that none of the six reach the model.

## Workflow

```
data_pipeline.py            load → drop leaked fields → build input text → date split → derive labels
        │
severity_model.py           train TF-IDF + LogReg on 10 grades → derive bucket + triage score
        │                   → pick cutoff on Sep → calibrate on Sep → evaluate on Oct → save
        │
severity_model_extended.py  sponsor comparison, operating-point menu, per-day recall@k,
        │                   subgroup breakdown with confidence intervals, explanations
        │
error_analysis.py           lists the harmed October reports the triage cutoff misses
        │
rolling_evaluation.py       retrains month by month (test Mar–Oct) to score 303 serious
        │                   cases instead of October's 20; a development metric
        │
prefill_extra_fields.py     HPI Designation classifier (pre-fill helper)
```

### How the model works

1. **Text to features:** TF-IDF on unigrams and bigrams, top 20,000 features,
   English stop words removed. It uses a custom token pattern so hyphenated
   terms like "X-ray" stay intact.
2. **Classifier:** multinomial logistic regression on all 10 grades, with
   `class_weight="balanced"` to handle the heavy imbalance.
3. **Derived outputs:** the 3-bucket label sums each group's grade
   probabilities. The triage score is the summed probability of grades E–I.
4. **Cutoff:** picked on September (validation) as the highest cutoff that
   still reaches ≥95% recall on harm cases, then frozen and applied to
   October unchanged. The 95% target is this project's own default, not a
   documented sponsor requirement.
5. **Calibration:** Platt scaling fit on validation, so the triage score reads
   as a real probability despite training's inflated harm rate.

### Running it

Scripts need Python 3.11 with pandas, scikit-learn and joblib (transformer
experiments also need torch, transformers, datasets and accelerate). The
dataset isn't in the repo; point `DATA_PATH` in `data_pipeline.py` at your
copy of the parquet file.

```bash
python3 severity_model.py            # trains, evaluates, saves to severity_model/
python3 severity_model_extended.py   # extended evaluation
python3 prefill_extra_fields.py      # HPI Designation classifier
python3 error_analysis.py            # reviews the harmed test reports the model misses
python3 rolling_evaluation.py        # month-by-month evaluation of the serious tier (a few minutes)
python3 predict.py reports.csv       # scores new reports with the saved model
```

`predict.py` takes a CSV or parquet file with the intake columns listed above
and writes one row per report: letter grade, 3-bucket label, raw and
calibrated triage score, and the priority-review flag. Use `-o` to choose the
output file (default `<input name>_predictions.csv`).

Retired and one-off experiment scripts live in `experiments/` (see its
[README](experiments/README.md)). Run them as modules from the repo root, for
example `python3 -m experiments.baseline_grader`.

### Tests

```bash
python3 -m pytest
```

Needs `pytest` installed. The tests check the pieces where a bug would give
wrong results without crashing:

- **Leakage guard:** each excluded column is rejected before it can reach the
  model input.
- **Date split:** `Event No.` prefixes decode to the right split, and
  malformed IDs raise an error.
- **Grade → bucket mapping:** all ten grades are in order and map to the right
  bucket.
- **Probability sums:** the bucket probabilities sum to 1, and the triage
  score equals `some` + `serious`.
- **Rolling folds:** each fold trains only on months before its cutoff and
  test months, no report is tested twice, and the last fold is the production
  split.
- **Saved model:** the files in `severity_model/` load from disk and score
  reports through `predict.py`, and post-review columns in the input don't
  change the scores.

None of the tests read the dataset. The saved-model tests are skipped until
`severity_model.py` has been run, because the `.joblib` files aren't in the
repo.

Outputs in `severity_model/`: `evaluation.json`, `extended_evaluation.json`,
`rolling_evaluation.json` and `config.json` are committed; the `.joblib` model files are gitignored.

## Results

All numbers are on the October test set (8,312 reports), from
`severity_model/evaluation.json` and `extended_evaluation.json`.

### Triage (harmed vs. not harmed)

At the production cutoff (0.236):

| Metric | Value |
|---|---|
| Recall | **96.0%** |
| Precision | 28.9% |

### Letter grade (10-way)

Exact accuracy 76.6%, mean absolute error 0.39 grades, quadratic-weighted
kappa 0.72.

### 3-bucket (derived)

| Bucket | Precision | Recall | Test cases |
|---|---|---|---|
| `none` | 0.99 | 0.99 | 7,909 |
| `some` | 0.76 | 0.83 | 383 |
| `serious` | 0.62 | 0.75 | 20 |

Macro F1: 0.82.

### Serious tier, rolling evaluation (development metric)

October has only 20 G/H/I cases, so `rolling_evaluation.py` repeats the
production split month by month: train on all earlier months, pick the cutoff
on the next month, test on the month after. Testing March through October
scores **303 serious cases**, each by a model that never saw its month. The
last fold is the production split and reproduces its cutoff. Numbers are from
`severity_model/rolling_evaluation.json`.

| G/H/I cases | Flagged by triage cutoff | Labelled `serious` |
|---|---|---|
| All, Mar–Oct | 297 / 303 = 98.0% (95% CI 95.7–99.1%) | 157 / 303 = 51.8% (46.2–57.4%) |
| Grade G | 68 / 69 = 98.6% | 14 / 69 = 20.3% (12.5–31.2%) |
| Grade H | 109 / 112 = 97.3% | 75 / 112 = 67.0% (57.8–75.0%) |
| Grade I | 120 / 122 = 98.4% | 68 / 122 = 55.7% (46.9–64.2%) |
| Mar–Aug only (oversampled months) | 263 / 267 = 98.5% (96.2–99.4%) | 133 / 267 = 49.8% |
| Sep–Oct only (natural harm rate) | 34 / 36 = 94.4% (81.9–98.5%) | 24 / 36 = 66.7% |

How to read it:

- **Use it to compare models, not as a headline.** January–August informed
  every modelling choice, so October alone is still the final test.
- **The triage cutoff is close to its ceiling on this tier** (6 misses in
  303), so it can't show much improvement. The `serious` label (146 misses)
  and the share of serious cases caught at a 5% false-alarm rate
  (276 / 303 = 91.1%) leave more room.
- **The `serious` label is the weak spot, grade G most of all.** Four of five
  permanent-harm reports are flagged for review but not labelled `serious`.
- **The `serious` label needs training data.** It labels 28% of serious cases
  correctly in the March fold (trained on January alone) and 53–75% in the
  last five folds, so the pooled 51.8% understates the production model.
- **Precision and flag rate are not pooled.** January–August oversamples harm
  (about 15% of reports versus 4–5% in September and October), which inflates
  both; the per-fold values are in the JSON.
- **March–August may be slightly easier than a natural month.** 98.5% versus
  94.4% flagged, but with 36 natural-rate cases the intervals overlap and the
  gap can't be confirmed.

### Compared with the sponsor's model

The sponsor reported sensitivity (the same thing as recall) at a fixed
specificity: the share of unharmed reports correctly *not* flagged. Setting
this model to the same 96.0% specificity:

| | Sensitivity (recall) | Specificity |
|---|---|---|
| Sponsor's Clinical-Longformer | 95.2% | 96.0% |
| This model | 90.3% | 96.0% |

This is not a like-for-like comparison. The sponsor's script feeds
`MANAGER COMMENTS` and `UNIT_ACTIONS_TAKEN` into the model, uses a random
rather than date-based split, and was measured on real data, not this
synthetic set.

## Known issues and limitations

- **Some serious cases are missed.** 2–3 of the 20 G/H/I test cases fall below
  the production cutoff. Their narratives use hedged, near-miss language
  ("could have received an incorrect dose") rather than outcome words.
  A cutoff of about 0.152 catches all of them, at roughly 25 more false
  alarms per day. That cutoff was fit on test, so it's optimistic.
- **Validation can't tune for those misses.** Every September G/H/I case
  scores ≥0.350, while October's misses score 0.152–0.310.
- **The serious tier is tiny.** 20 test cases (6 G, 14 H, 0 I) and 16 validation
  cases, so `serious` metrics have wide confidence intervals and grade I isn't
  evaluated at all. The rolling evaluation above covers 303 cases (122 grade
  I) as a development metric.
- **Rare grades are weak on their own.** B1 F1 is 0.21 and G F1 is 0.55. The
  letter-grade macro F1 is 0.59, versus 0.82 once grades are grouped into
  buckets.
- **Precision is low at the default cutoff.** About 7 in 10 flagged reports are
  false alarms. That's the cost of the 95% recall target.
- **Narrative/label mismatch.** Many false negatives say "no harm noted" while
  carrying a harm grade. This could be real, or noise from the synthetic data.
- **Synthetic data only.** No real-data validation. Public alternatives were
  checked: NRLS/LFPSE and SAFRON are access-gated, FAERS has no narrative text,
  and IFMIR has no severity labels.
- **No template-overlap check.** There's no archetype ID column, so near-duplicate
  templates across train and test can't be ruled out.
- **Subgroup results are inconclusive.** Recall intervals overlap across all
  services, and subgroup membership is generator-assigned, not real.
- **Some fields aren't modeled.** Level of Investigation isn't modeled because 98%
  of rows have one value, which leaves too few RCA cases.

## Open tasks

### Code improvements

- [x] **Add a prediction entry point.** `predict.py` loads `severity_model/`
      and scores a CSV/parquet file, writing out grade, bucket, calibrated
      triage probability and the flag.
- [x] **Add tests for the safety-critical pieces.** `tests/` has `pytest`
      checks for the excluded-column assertion, `Event No.` split decoding,
      the grade → bucket mapping, `bucket_probs` summing to 1, and a
      round-trip load of the saved model.
- [x] **Separate production code from experiments.** `baseline_grader.py`,
      `ordinal_vs_coarse.py`, `combined_mtl.py`, `augment_rare_grades.py`,
      `finetune_distilbert.py`, `leakage_inflation_check.py` and
      `longformer_colab.ipynb` now live in
      `experiments/`, so it's obvious that `data_pipeline.py`,
      `severity_model.py` and `severity_model_extended.py` are the pipeline.
- [x] **Point `error_analysis.py` at the production model.** It was written
      for the 3-bucket DistilBERT output. It now loads the saved 10-grade
      model and reviews the harmed reports the triage cutoff misses, so the
      false-negative review is repeatable on what's shipped. It moved back
      to the repo root for that reason.

### Experiments

- [ ] **Tune DistilBERT's `MAX_LENGTH`.** 128 tokens was a guess, never
      checked against the dataset's token-length distribution. The 0.79
      result likely undersells it.
- [ ] **Train DistilBERT on all 10 grades.** It's only been run on 3 buckets.
      Deriving bucket and triage outputs the same way as production would give a
      like-for-like comparison. Its better `serious` precision makes this worth
      doing.
- [ ] **Run the sponsor's Clinical-Longformer script head-to-head.** Needs one
      GPU/Colab session. Run it twice on the date split: once as written, and
      once with `MANAGER COMMENTS` / `UNIT_ACTIONS_TAKEN` removed. It will
      need updates for current `transformers`:
      - `compute_loss` needs `num_items_in_batch=None`
      - `evaluation_strategy` → `eval_strategy`
      - `tokenizer=` → `processing_class=`

      This measures the leakage effect on synthetic data only.
- [ ] **Retry the combined model on a fine-tuned encoder.** Use DistilBERT or
      Bio_ClinicalBERT, jointly fine-tuned for event type and severity. The
      frozen-MiniLM version was too weak to show whether the event-type clue
      helps a stronger model.
- [x] **Target the "quiet" serious misses.** Tested on the rolling evaluation
      (`experiments/quiet_serious_misses.py`); nothing adopted. Hedge-phrase
      features and high-risk word rules cost more false alarms than they
      recover, and the rules do no better than lowering the cutoff. A separate
      serious-case detector recovered 5 of the 6 misses but flags 22% of
      unharmed reports instead of 14%, and its edge over a lowered cutoff
      (3 cases) isn't statistically reliable. Still open: whether these cases
      are real or label noise (see below).
- [x] **Cluster drug names.** Tested (`experiments/drug_name_groups.py`); no
      effect. Hand-mapped drug classes and context-based clusters each change
      at most 3 of 303 serious cases on any measure.
- [ ] **Keep negations and modals in the vocabulary.** The vectorizer's stop
      list removes "no", "not", "could" and "would". Keeping them left hurt
      recall unchanged (94.8% → 95.0%) while flagging fewer unharmed reports
      (14.1% → 12.5%) in the rolling evaluation. A side finding of the
      quiet-misses experiment, not yet applied to production or checked on
      October.

### Needs sponsor input

- [ ] **Triage cutoff policy:** 0.236 (misses 2–3 serious cases) or about 0.152
      (catches all, with about 25 more false alarms per day). This is a
      staffing and risk decision.
- [ ] **Are the missed serious cases real or label noise?** Raise this at the
      Safety Officer shadowing session.
- [ ] **Narrative/label mismatch:** the same question at larger scale, for reports
      that say "no harm noted" but carry a harm grade.
- [ ] **Archetype IDs**, needed to check template overlap between train and test.
- [ ] **Is Event Type filled in at intake?** If so, the `11-buckets` 11-way
      classifier predicts a field that already exists (see end of
      `PIPELINE.md`).
