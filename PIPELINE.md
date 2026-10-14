# Pipeline architecture

Two separate pipelines, built by two people, answering two different
questions from the sponsor's requirements. They never share a model — they
combine only at the very end, in the final per-report record.

- **Severity** ("how bad is this report") — this repo.
- **Event type** ("what kind of report is it") — the teammate's `11-buckets`
  repo, answering the sponsor's "group reports into recurring safety themes"
  requirement.

File locations: `baseline_grader.py`, `ordinal_vs_coarse.py`,
`combined_mtl.py`, `augment_rare_grades.py`, `finetune_distilbert.py`,
`leakage_inflation_check.py` and `longformer_colab.ipynb` are named below
without a path; they now live in `experiments/`.

---

## Severity pipeline

```mermaid
flowchart TD
    A["Report\n23 raw columns"] --> B["Keep only submission-time fields:\nevent_comments, Location Name,\nEncounter Service, Age at Encounter,\nME/ADR medication fields\n\nBlocked: manager_comments,\nunit_actions_taken, shareable_lessons"]
    B --> C["Glue into one string:\nUnit: ... Service: ... Age: ...\n+ narrative"]
    C --> D["Text -> numbers: TF-IDF\n20,000 words/phrases,\nunigrams + bigrams"]
    D --> E["Train on 50,091 reports\n(Jan-Aug), class-weighted\nLogistic Regression"]
    E --> F{"One probability\nper grade"}
    F -->|"highest single grade"| G["Pre-fill letter grade"]
    F -->|"sum of E..I"| H["Triage risk score"]
    E --> I["Checked on Sep + Oct\n(never trained on)"]
```

### Example, end to end

Input fields:

```
Unit: ICU. Service: Cardiology. Age: 74. Prescribed: Warfarin 5mg.
Patient found on floor at 0300, alert and oriented, complaining of
left hip pain. X-ray confirmed fracture. Transferred to OR for repair.
```

TF-IDF, current production code (`severity_model.py`; the retired
`baseline_grader.py`/`ordinal_vs_coarse.py` use the same shape without the
`token_pattern` fix below, since they predate it):

```python
TfidfVectorizer(
    stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=3,
    token_pattern=r"(?u)\b[\w-]+\b",  # keeps hyphenated terms like "X-ray" intact
)
LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
```

Actual output from the current retrained model on this exact report (not
illustrative -- run through `severity_model/model.joblib`):
`A:3% B1:5% B2:6% C:15% D:9% E:36% F:18% G:5% H:1% I:1%`
→ pre-fill = **E** (highest), triage score = **61%** (sum of E through I).

### The three branches that exist in the code today

All three start from the identical steps above (columns → glue → TF-IDF).
They differ only in what "correct answer" column each one is trained
against. They are three separate trained models, not one pipeline with
three outputs.

| Branch | Script | Trained against | Why it exists |
|---|---|---|---|
| 1 — 3 buckets | `baseline_grader.py` | `none` / `some` / `serious` | Original version of this project |
| 2 — 10 grades | `ordinal_vs_coarse.py` | `A, B1, B2, C, D, E, F, G, H, I` | Sponsor needs the real PSRS letter, not a bucket |
| 3 — binary | `ordinal_vs_coarse.py` | `hurt` / `not hurt` | Built once, only to check whether Branch 2's derived triage score is as good as a model built just for triage. Not a production model. |

**Decision: Branch 2 is the one to keep — done, 2026-09-24.** `severity_model.py`
is the finished version: trains the 10-grade model, derives the letter grade,
the 3-bucket label (sum A+B1+B2+C+D / E+F / G+H+I), and the triage score all
from one model, picks a triage cutoff on validation for this project's own
default target of >=95% recall (not a documented sponsor requirement),
calibrates the triage probability (Platt scaling, to correct for training
data oversampling harm relative to the real rate), and
saves everything to `severity_model/` (`vectorizer.joblib`, `model.joblib`,
`calibrator.joblib`, `config.json`, `evaluation.json`). `baseline_grader.py`
(Branch 1) and the standalone binary model in `ordinal_vs_coarse.py`
(Branch 3) are both superseded -- kept in the repo for history, not run
going forward.

### Results, October test set (8,312 reports)

**Branch 1 — 3 buckets:**

| Bucket | Precision | Recall |
|---|---|---|
| none | 0.99 | 0.96 |
| some | 0.55 | 0.89 |
| serious | 0.55 | 0.80 |

Accuracy 96%, macro F1 0.77.

**Branch 2 — 10 grades** (all numbers below read directly from
`severity_model/evaluation.json`, current as of the round-4 retrain --
this table and the "Final, saved model" section below are the pipeline's
canonical numbers; the Severity Model Reference doc mirrors them):

| Grade | Precision | Recall | Test rows |
|---|---|---|---|
| A | 0.45 | 0.68 | 442 |
| B1 | 0.15 | 0.34 | 47 |
| B2 | 0.75 | 0.81 | 1,505 |
| C | 0.77 | 0.70 | 2,487 |
| D | 0.89 | 0.81 | 3,428 |
| E | 0.58 | 0.84 | 318 |
| F | 0.65 | 0.86 | 65 |
| G | 0.60 | 0.50 | 6 |
| H | 0.72 | 0.93 | 14 |
| I | — | — | 0 (no test cases at all) |

| | This model | Guess-most-common-grade |
|---|---|---|
| Exact-letter accuracy | 76.6% | 41.2% |
| Mean grade-distance error | 0.39 | 0.95 |
| Quadratic-weighted kappa | 0.72 | 0.00 |

### Final, saved model (`severity_model.py`) — closed out 2026-09-24

**3-bucket, derived from the 10-grade model** (replaces Branch 1 outright):

| Bucket | Precision | Recall | F1 |
|---|---|---|---|
| none | 0.99 | 0.99 | 0.99 |
| some | 0.76 | 0.83 | 0.79 |
| serious | 0.62 | 0.75 | 0.68 |

**Honest trade-off vs. the original Branch 1 model** (which trained directly
on 3 buckets, no letter grade): precision on `some`/`serious` went up a lot
(0.55 → 0.76 / 0.62), but recall went down (`some` 0.89 → 0.83, `serious`
0.80 → 0.75). This project's default priority is high sensitivity even at
the cost of precision — so a straight argmax over the derived buckets is
*not* automatically the right cutoff to use if the 3-bucket label itself is
still shown to reviewers. The **calibrated triage score below is the
answer to that priority**, tuned explicitly for recall; the 3-bucket label
is a secondary, human-readable summary, not the flagging mechanism.

**Triage cutoff, picked on validation only for >=95% recall, frozen and
applied to test:**

| | Value |
|---|---|
| Cutoff (raw triage score) | 0.2361 |
| Recall achieved on validation | 95.2% |
| Recall on test (untouched, same cutoff) | **96.0%** |
| Precision on test | 28.9% |
| Share of all test reports flagged | 16.1% |
| AUPRC (cutoff-independent ranking quality) | 0.871 |

28.9% precision at 96.0% recall is the honest cost of this project's own
default priority: catching nearly all real harm means accepting that roughly
3 in 4 flagged reports won't turn out to be harmful. That's an explicit,
measured trade-off now, not a guess. See the Severity Model Reference §14.2
for the fuller operating-point menu, including the trade-off for catching
every "serious" case specifically.

**Calibration** (Platt scaling on validation, to correct for the training
set's oversampled harm rate vs. the real rate in val/test): Brier score
0.031 (raw) → 0.014 (calibrated) — the calibrated probability is
meaningfully more trustworthy as an actual risk percentage, not just a
ranking signal.

**Known gap, not fixable by modeling:** `I` (death) has zero test-set rows;
`G` (permanent harm) has 6. No model can be validated on death cases with
this split — worth raising with the sponsor directly.

**Saved artifacts:** `severity_model/vectorizer.joblib`, `model.joblib`,
`calibrator.joblib`, `config.json` (the frozen cutoff), `evaluation.json`
(this whole report, machine-readable). Verified end to end: reloading these
four files from disk and predicting on a new report (not used in training)
reproduces the expected grade, bucket, and flag decision.

### Sponsor-facing review, round 1 (2026-09-25)

A detailed review raised six concrete gaps, addressed as far as possible
without sponsor input. `leakage_inflation_check.py`, `severity_model_extended.py`,
and `prefill_extra_fields.py` were added; `data_pipeline.py` now handles a
`Deleted` harm score gracefully and cites the sponsor's own code for the
B1-before-B2 grade ordering.

### Sponsor-facing review, round 2 (2026-09-25) — two factual corrections, several bugs found and fixed

The round-1 writeup had real errors, caught in a second review:

- **Wrong sponsor number.** The sponsor reported 95.2% sensitivity at 96.0%
  specificity, not "96% recall." Corrected comparison at matching
  specificity: this model gets 90.3% sensitivity at 96.0% specificity,
  trailing their reported number. AUROC 0.986 added as the threshold-free
  comparison point.
- **"No access to their code" was disputed, then re-confirmed as still true.**
  *(Corrected in round 3: this was wrong. The script existed on disk the
  whole time, in `~/Downloads/Capstone Engineering Files/` -- the check here
  only searched conversation history, not the filesystem. Don't lift this
  bullet out of context as a factual claim.)* Original text: a review
  claimed their training script was "attached earlier in this conversation"
  -- it was not; no such file exists anywhere in this project's history. The
  scorecard now says the head-to-head is blocked on the script itself, not
  compute, and will be revisited if that script is ever actually shared.
- **Tokenizer bug, fixed, model retrained.** scikit-learn's default token
  pattern dropped single characters, so "X-ray" tokenized to "ray" --
  wrong in every downstream explanation, not just the demo. Fixed
  (`token_pattern=r"(?u)\b[\w-]+\b"`) and the production model, the leakage
  check, and all evaluations retrained/rerun on the corrected vectorizer.
- **Recall@k was structurally misleading.** Pooled over the full 31-day test
  set, "top 20 catches 5%" looked bad but meant nothing -- recomputed per
  day and averaged: top 20 catches 89.3%, top 50 catches 97.1% of each
  day's harm cases. A completely different, correct number.
- **Subgroup table now carries n_pos and Wilson 95% confidence intervals** --
  the earlier version's "Obstetrics: 0.769 recall" looked like a finding;
  with intervals shown, every service's recall interval overlaps every
  other's. No confirmed subgroup disparity survives. Subgroup membership was
  also generator-assigned, not real -- the analysis that counts is the one
  on the real container run.
- **Report Type is very likely circular with the harm label.** "Serious
  Event" agrees with the true hurt (E-I) label 99.3% of the time. If this
  field is set after review (plausible given that correlation), pre-filling
  it from a model has the same circularity the original leakage bug had.
  Needs sponsor confirmation -- open question, not resolved here.
- **Leakage-effect claim narrowed.** AUPRC/AUROC deltas from adding the two
  blocked fields are near zero on this synthetic dataset -- the finding is
  now stated as "little effect here," not extrapolated to the sponsor's real
  data, where manager comments likely carry more signal.

Full writeup, all numbers, and the automation-bias mitigation plan: see the
Severity Model Reference doc (ask for the link if you don't have it).

**Not yet done:** committing and tagging the commit that produced the
evaluation JSONs -- needed for these numbers to be reproducible, but a
deliberate choice left to you, not done automatically.

### Sponsor-facing review, round 3 (2026-09-25) -- the script exists, and it confirms leakage

Round 2's "no such file exists anywhere in this project's history" was true
of the conversation history, but wrong about the filesystem: the script was
sitting in `~/Downloads/Capstone Engineering Files/`. Reading it changed two
things and fixed the doc's remaining stale numbers:

- **Their script confirms leakage directly -- no synthetic-data proxy needed
  for this part.** `LLM_fine_tuning_MIDA_PSRS_score_Weights_clean.py`
  (`yikuan8/Clinical-Longformer`, 4096-token max length) builds its model
  input with a function that includes `MANAGER COMMENTS` and
  `UNIT_ACTIONS_TAKEN` verbatim -- the same two post-investigation fields
  this pipeline excludes by design. That's a fact about their code, not an
  inference. Their split is also random-stratified, not date-based, which is
  a second reason their number and this model's number aren't directly
  comparable. The scorecard now reads "pending GPU run," not "blocked" --
  next step is training their script twice on a Colab T4 (once as written,
  once with those two fields stripped) *(corrected in round 4: that rerun
  can only use synthetic data, since that's what this project has access to
  -- it measures the leakage effect on synthetic data, not "the effect on
  real data." Their reported number was measured on the real CY2024 Excel
  export and only someone with access to that data can actually test it)*.
- **The two systematic "serious" misses got pulled and read, not just
  counted.** Across every recall target tested, the same 1-2 of the test
  set's 20 G/H/I cases are missed regardless of cutoff
  (`SYNPROD-TEST-0006980`, true grade H, score 0.208; `SYNPROD-TEST-0009928`,
  true grade G, score 0.152). Both reports narrate the moment of discovery
  ("could have received an incorrect... dose," "patient looked fine and was
  talking normally") -- the eventual G/H severity was determined later, by a
  reviewer who knew the outcome, using information this text field never
  contained. *(Corrected in round 4, and this is the important one: "no
  threshold change fixes this" was false -- a cutoff around 0.152 does catch
  both, at a real, computable cost in flag rate. And "the reviewer knew the
  outcome later" was an unverified guess presented as fact; round 4 restates
  it as one of two hypotheses, the other being synthetic-data label noise.
  Don't lift either claim out of context.)*
- **Stale numbers from the pre-retrain model, fixed.** The doc's §8/§10/§11
  tables still showed the pre-tokenizer-fix numbers (96.8% recall, 27.5%
  precision, 76.7% accuracy, cutoff 0.225, 17.0% flag rate) even though §14
  already had the retrained ones. All now read from `evaluation.json`
  directly: 96.0% recall, 28.9% precision, 76.6% accuracy, cutoff 0.2361,
  16.1% flag rate.
- **Cross-reference fixed:** §12's prior-shift pointer now says §14.4, not
  §14.3 (an earlier renumbering had drifted).
- **Revision-history language stripped from the sponsor-facing doc itself**
  (the "first version over-claimed" / "built across two rounds of review"
  framing). That belongs here, not in front of the sponsor -- their copy now
  states current findings only, keeping bug notes (tokenizer fix, Laplace
  smoothing) only where they explain a design choice.

**Bottom line:** the round-2 "Bottom line" is met. The only work still
genuinely open is the GPU rerun of their actual script (now unblocked,
just not yet done), the archetype-overlap check (needs sponsor-provided
IDs), and the Report Type intake-vs-post-review question (needs a sponsor
answer, not resolvable from either script).

### Sponsor-facing review, round 4 (2026-09-25) -- the missed-cases fix, and stale numbers that survived two rounds

Round 3's headline claim was wrong, caught before it shipped:

- **"No threshold change fixes this" was false.** The two lowest-scoring
  missed G/H cases score 0.208 and 0.152 -- a cutoff of ~0.152 catches both,
  plus a third case (`SYNPROD-TEST-0001789`, grade G, score 0.310) that's
  also missed at every recall target below 95%. Catching all 20 serious
  cases in the test set costs a flag rate of 28.9% (vs. 16.1% today) --
  roughly 25 more false alarms/day at the sponsor's stated volume. Added as
  an explicit row in the operating-point menu (§14.2) instead of asserting
  the trade-off doesn't exist. That's the sponsor's call, not this
  pipeline's to make silently.
- **The "reviewer knew the outcome later" explanation was a guess presented
  as fact.** Nobody verified it. These are LLM-written synthetic narratives
  generated from an archetype and a label -- "patient looked fine, talking
  normally" paired with permanent harm is at least as likely to be generator
  label noise as a real delayed-harm pattern. Both hypotheses are now stated
  side by side, with a concrete question for the Safety Officer shadowing
  session: do real G/H reports ever read like a caught near-miss at intake?
- **A cleaner fix than reopening leakage, added:** re-triage on update.
  Manager comments are illegitimate for initial triage but perfectly
  legitimate once they actually exist -- a report that scored low at intake
  and later gets a worrying manager comment gets re-scored and re-flagged at
  that point. Real feature, not leakage, and it directly targets this
  failure mode.
- **§3's TF-IDF snippet was missing the token_pattern fix** -- code shown
  didn't match the code that actually ran. Fixed.
- **§11's per-grade and 3-bucket tables were carried over from the
  pre-retrain model**, identical to two decimals even after accuracy moved
  in round 3. Recomputed directly from `evaluation.json`: B1 recall
  0.32->0.34, E recall 0.85->0.84, F precision/recall 0.62/0.85->0.65/0.86,
  H precision 0.81->0.72, MAE 0.40->0.39, kappa 0.71->0.72, serious-bucket
  precision/F1 0.65/0.70->0.62/0.68.
- **§14.9 had a note to me, not the sponsor** ("this is the user's call to
  make") -- removed; the section now just states that the evaluation JSONs
  are committed and tagged.
- **§14.1 reframed.** "Confirmed leakage in their pipeline" was accurate but
  is the sponsor's own code -- raised verbally at the weekly meeting first,
  and the doc now reads "their script includes two post-review fields as
  input; we'll measure the effect." Also added the point round 3 missed
  entirely: their 95.2%/96.0% was measured on the real CY2024 Excel export.
  A Colab rerun here can only run on the synthetic dataset, so it measures
  the leakage effect on synthetic data for Longformer -- it cannot reproduce
  or refute their real-data number. Only someone with access to the real
  data can actually test that number.

**Heads-up for whoever runs the Colab rerun:** their script will likely
break as-is on current `transformers` -- `compute_loss` needs a
`num_items_in_batch=None` kwarg (4.46+), `evaluation_strategy` is now
`eval_strategy`, and `tokenizer=` in `Trainer(...)` is now
`processing_class=`. Also, their `make_report()` writes the literal string
`"nan"` into the text for empty fields (via unguarded f-string formatting of
`NaN` floats) -- harmless, but this pipeline's `build_input_text` doesn't do
that, so note the difference if scores are compared field-for-field.

Committed and tagged tonight per the round-4 review's instruction --
`severity_model/evaluation.json` and `extended_evaluation.json` are pinned
to the commit that produced them.

### Sponsor-facing review, round 5 (2026-09-25) -- the "catch every serious case" row was fit on test, and PIPELINE.md itself needed cleanup

Two more issues in the doc, plus this file catching up to what it was
describing:

- **The round-4 "catch every serious case" row picked its cutoff on the test
  set, same mistake in spirit as tuning on data you're reporting on.** Every
  other row in the operating-point menu freezes its cutoff on validation;
  this one didn't. Fixed by deriving it on validation instead: September has
  only 16 G/H/I cases (itself a limitation), and the cutoff that catches all
  16 of *those* still misses 3 of October's 20 when frozen and applied to
  test. The test-fit cutoff (0.152, catches all 20) is kept for contrast
  only, explicitly labeled optimistic-by-construction, with a Wilson 95%
  interval on "20/20" (~[0.84, 1.00]) stated so it isn't read as a guarantee.
- **Two leftover notes still contradicted the softened framing**: the §14
  intro's "confirmed leakage source in the sponsor's own script," and a
  literal note-to-self in §14.1 about raising something at the weekly
  meeting before the section ships. Both cut.
- **§11 repeated the same unsupported claim §8 had already been corrected
  for** -- "the sponsor's stated priority is recall." Sponsor priority was
  never actually documented; changed to "this project's default priority,"
  same fix applied to §8 in an earlier round.
- **This file itself was stale and hadn't caught up to its own changelog.**
  The "Decision" and "Final, saved model" sections still showed the
  pre-round-3 numbers (cutoff 0.225, recall 96.8%, precision 27.5%, flag
  rate 17.0%, the old per-grade table) even though the changelog below
  documented the fix. Regenerated directly from `evaluation.json`. The
  TF-IDF snippet near the top didn't show `token_pattern` either -- fixed.
  The two false claims preserved in the round 2 and round 3 entries above
  (file doesn't exist; no threshold fixes the misses) now have inline
  `(corrected in round N)` notes so they can't be quoted out of context.
  The merge-rejection paragraph now states plainly that its 0.77 baseline is
  the old Branch 1 model, not current production (0.82). The 83.1%/87.3%
  event-type comparison is now called a 4-point gap ("same ballpark"), not
  "confirmed." The running example's "Grade F" in the closing table was
  swapped for the actual model's real output on that text (Grade E, 61%
  triage score) to match the Severity Model Reference doc's own example.
- **Added the hurt-detection head-to-head and the Event Type open question**
  (both below) at the user's request, to make the severity-approach case
  concrete and to flag the 11-way classifier's possible redundancy with an
  intake-filled field before the teammate invests further in it.

### Sponsor-facing review, round 6 (2026-09-25) -- round 5's new row was itself wrong

*(Corrected in round 6: round 5's "catch every val G/H/I case" row, described
above, was logically backwards and has been removed from the doc.)* The
val-derived cutoff (0.350) is *higher* than the 95%-target cutoff already in
the menu (0.236) -- since a lower cutoff flags a superset of what a higher
one flags, the 95% row already caught every September serious case for
free, and the new row was strictly worse on October (misses 3 vs. the
95% row's 2). "Fit on validation to catch the serious tier" here just picks
the highest cutoff that still happens to clear September's cases -- the
least sensitive choice, not an optimization.

The real finding, kept in place of the row: every September (validation)
G/H/I case scores at least 0.350, well above any cutoff actually in use,
while October's misses score 0.152-0.310 -- below any September serious
case. Validation tuning can't help here because September has no serious
case that looks like October's hard ones. The two months also have a
different serious-tier mix (September: 0 H, 9 I, 7 G; October: 14 H, 0 I,
6 G) -- ordinary variance in a ~20-case-a-month tier, or more weight behind
the label-noise hypothesis. The test-fit 0.152 contrast row is unchanged.

Also fixed: the scorecard's kappa (0.71 -> 0.72, matching §11) and the
derived-3-bucket macro F1 cited in the "Alternatives" table (0.83 -> 0.82,
matching this file and `evaluation.json`).

### Tested and rejected: hand-written synthetic augmentation for G/H/I (2026-09-25)

Train has only 71 G, 130 H, 158 I rows out of 50,091 -- tried adding more
synthetic examples for exactly these three grades to see if it helps.
`augment_rare_grades.py` hand-writes 60 new reports (20 each) across 15
distinct clinical scenarios (medication overdose, wrong-site surgery, missed
sepsis, transfusion reaction, restraint injury, anesthesia complication,
equipment failure, pressure injury, delayed diagnosis, fall with head
injury), appended only to the training split -- validation and test are
untouched, so the comparison is fair.

**Result: no measurable effect.** Accuracy, MAE, and kappa moved by noise
(+0.0016 / -0.0011 / -0.0003). G and H precision/recall were identical to
three decimals before and after. Both of the two systematic test-set misses
(§14.2 in the Severity Model Reference) were still missed, at nearly
identical scores (0.208->0.200, 0.152->0.158).

**Traced the mechanism, not just the null result:** checked which of the 268
new vocabulary words the augmented training introduced actually appear in
the two missed real test reports. Answer: zero words for one, one word
("critically") for the other. The two real misses use quiet, hedged language
("could have received an incorrect dose," "patient looked fine and was
talking normally") -- the same finding from the Severity Model Reference
§14.2. My hand-written synthetic examples used explicit, unambiguous outcome
language ("cardiac arrest," "resuscitation unsuccessful," "confirmed
permanent"), because that's how I write about death and permanent harm.
**More synthetic data of the obvious kind cannot teach the model to
recognize the quiet kind** -- it doesn't share any vocabulary with the cases
that are actually hard. Any future attempt at this would need synthetic
examples that specifically mimic the hedged, near-miss-sounding style, which
is a much harder (and more suspect) thing to hand-write convincingly, given
this project can't yet tell whether that style is a real reporting pattern
or synthetic-label noise (§14.2 again).

Rejected, same treatment as the MiniLM merge and prior-shift correction
experiments: a real result, documented, not deployed.

### Report Type: found real evidence, not just a hypothesis (2026-09-25)

The earlier finding (§14.7 in the Severity Model Reference) was "99.3%
agreement with the hurt label, needs sponsor confirmation whether it's set
at intake or after review." Two things found by looking directly at the
dataset's own columns and values push this further, though sponsor
confirmation is still the only way to fully close it:

- **The column is literally named `Analyst-Report Type*`.** Compare to
  `Event Type`, the separate, much more detailed field the 11-buckets
  pipeline uses (specific codes like "E - Lab test problem - mislabeled
  specimen") -- that one has no "Analyst-" prefix and reads as
  reporter-filled at intake. The "Analyst-" prefix on Report Type is a
  real, concrete signal that this field is assigned by a safety analyst,
  not the frontline staff member filing the report -- which is exactly the
  post-review assignment pattern that would make it circular with the
  harm label, same as the original leakage fields.
- **A closer look at what's actually inside "Serious Event" complicates the
  original 99.3% number.** Of 8,113 rows labeled "Serious Event," 6,300 are
  grade E and 918 are grade F -- the softer end of the hurt range. Only 4
  are I, 3 are G, 1 is H. Most of the worst cases (147 of 167 I-grade rows,
  127 of 144 H-grade rows, 71 of 84 G-grade rows, across the whole dataset)
  are actually filed as "Incident," not "Serious Event." So the 99.3%
  agreement is real but is
  driven by volume in the E/F range -- "Serious Event" is not a reliable
  signal for the very worst outcomes specifically, which matters if anyone
  planned to use it as a proxy for the G/H/I tier rather than the broader
  hurt/not-hurt split.
- **The date-gap between "Event Date" and "Date Received" doesn't help**
  distinguish this -- median gap is 0 days for every Report Type category,
  so it only tells us when the report was logged, not when the analyst
  made the Report Type call (there's no timestamp for that step in this
  dataset).

Net: the column name is real, useful evidence toward "set after review, by
an analyst" -- stronger than an unresolved guess, still short of a
sponsor-confirmed fact. Recommend leading with this in the sponsor
conversation rather than opening with "we don't know."

**CONFIRMED, same day:** the user confirmed directly -- the analyst/reviewer
fills in Report Type after review, not the reporter at intake. The evidence
above was correct. This is no longer an open question.

**Action taken as a result:** Report Type is removed from `prefill_extra_fields.py`
as a modeled pre-fill target. `train_and_eval()` is no longer called for it,
`report_type_model.joblib` is no longer produced (the stale artifact was
deleted), and `check_report_type_as_proxy()` is now a diagnostic-only
function documenting the confirmed finding, not a prompt for sponsor
confirmation. The Severity Model Reference's scorecard and §14.7 are updated
to match -- Report Type is now in the same excluded category as
`manager_comments`/`unit_actions_taken`, just for the pre-fill task rather
than the severity model's input.

### DistilBERT and the MiniLM merge, actually re-run (2026-09-25)

Both were flagged for a long time as "not yet re-run on the current pipeline"
-- finally done, on the current leakage-safe, date-split data. Local
environment needed `datasets` and `accelerate` installed (missing from the
3.11 env; installed via pip, no other environment issues).

**DistilBERT, fine-tuned end-to-end on the 3-bucket label** (`finetune_distilbert.py`,
~44 min on Apple GPU/MPS): macro F1 **0.79** on the frozen October test set.

| | none | some | serious |
|---|---|---|---|
| Precision | 0.99 | 0.55 | 0.78 |
| Recall | 0.97 | 0.85 | 0.70 |

This is a real, mixed result, not a clean win or a clean loss: it beats the
old direct-3-bucket TF-IDF baseline (0.77) but falls short of the current
production model's derived-3-bucket macro F1 (0.82). More interesting than
the aggregate number is the trade: DistilBERT's precision on the rare
"serious" class (0.78) is much better than production's (0.62), but its
recall on that same class is lower (0.70 vs. 0.75), and its "some"-class
precision is notably worse (0.55 vs. 0.76). Since this project's stated
priority is recall over precision on the harmful classes, this trade cuts
the wrong way to justify a swap -- but it's close enough, and different
enough in shape, to be a real candidate, not a rejected one. **Caveat inherited
from the original build, not newly introduced:** `MAX_LENGTH=128` was always
a guess, never checked against this dataset's actual token-length
distribution -- a properly tuned sequence length could move this number
before any decision is made either way.

**MiniLM multi-task merge**, rerun with fresh (not cached) embeddings:
confirms the original rejection -- see the entry above this one. No change
in conclusion, only in confidence (this run ruled out "the old number was
just a stale-cache artifact").

Not committed yet; `distilbert_output/` is gitignored (768MB of checkpoints)
and `test_results.json` inside it is the only file worth keeping — not yet
copied anywhere durable.

### Rolling evaluation: the serious tier on 303 cases instead of 20 (2026-10-06)

Every claim about G/H/I so far rests on October's 20 cases, too few to tell
whether a change helps or hurts them. `rolling_evaluation.py` repeats the
production split shape with an expanding window: train on months up to k-2,
pick the >=95%-recall cutoff on month k-1, freeze it, test on month k. With
January as the smallest training set that is 8 folds testing March-October,
and every report in those months is scored once by a model that never saw its
month. The last fold is the production split, and the script asserts that it
reproduces the saved cutoff (0.2362).

Supporting changes: `data_pipeline.py` now carries `event_month` (from
`Event Date`, splitting only, never a feature), checks that months agree with
the `Event No.` split, and provides `rolling_folds()`. `severity_model.py`'s
training moved into `fit_model()` so each fold trains exactly what production
trains; `evaluation.json` and `config.json` were byte-identical after the
refactor.

Pooled over 303 serious cases (69 G, 112 H, 122 I -- the first evaluation of
grade I at all, since October has none):

| | Count | Rate | Wilson 95% |
|---|---|---|---|
| Flagged by the triage cutoff | 297 / 303 | 98.0% | 95.7-99.1% |
| Labelled `serious` by the derived bucket | 157 / 303 | 51.8% | 46.2-57.4% |
| ... grade G | 14 / 69 | 20.3% | 12.5-31.2% |
| ... grade H | 75 / 112 | 67.0% | 57.8-75.0% |
| ... grade I | 68 / 122 | 55.7% | 46.9-64.2% |
| Outscore 95% of the month's unharmed reports | 276 / 303 | 91.1% | 87.3-93.8% |

Per fold (`serious` label recall rises with training size; the triage flag
does not):

| Test month | Train reports | Cutoff | Serious flagged | Labelled `serious` |
|---|---|---|---|---|
| Mar | 6,383 | 0.283 | 45/46 | 13/46 (28%) |
| Apr | 12,702 | 0.268 | 47/47 | 18/47 (38%) |
| May | 19,280 | 0.221 | 29/29 | 15/29 (52%) |
| Jun | 25,722 | 0.210 | 37/39 | 26/39 (67%) |
| Jul | 32,006 | 0.251 | 57/58 | 31/58 (53%) |
| Aug | 37,779 | 0.240 | 48/48 | 30/48 (62%) |
| Sep | 43,536 | 0.251 | 16/16 | 9/16 (56%) |
| Oct | 50,091 | 0.236 | 18/20 | 15/20 (75%) |

What this does and does not show:

- **It is a development metric, not a second test set.** Jan-Aug informed
  every modelling choice. October alone remains the final check.
- **Triage flagging of serious cases is near its ceiling** (6 misses in 303),
  so that number cannot distinguish candidate models. The `serious` label
  (146 misses) and the catch rate at a fixed 5% false-alarm rate (27 misses)
  can. Later experiments should report those, paired on the same cases with
  `compare()`.
- **Grade G is rarely labelled `serious`** (20%), though almost always
  flagged. Not visible before with 6 October G cases.
- **Pooled `serious`-label recall understates production**, because early
  folds train on far less data.
- **Precision and flag rate are per-fold only.** Jan-Aug oversamples harm
  (14.6-15.8% of test-month reports vs. 4.1% and 4.8% in Sep and Oct), so
  precision there (46-58%) is not comparable to Sep/Oct (28-29%).
- **Are the oversampled months easier?** Serious flagged: 263/267 (98.5%,
  CI 96.2-99.4%) in Mar-Aug vs. 34/36 (94.4%, CI 81.9-98.5%) in Sep-Oct. The
  direction fits October's "quiet" misses being under-represented in Jan-Aug,
  but the intervals overlap; not established.
- **The 2 October misses match production** (18/20 flagged, 15/20 labelled
  `serious` = the 0.75 recall in `evaluation.json`).

### Tested and rejected: targeting the "quiet" serious misses (2026-10-13)

`quiet_serious_misses.py`, on the rolling evaluation, each variant paired
against the production model on the same 303 serious cases. Results in
`experiments/quiet_serious_misses_results.json`.

Limits on what this could show: only 6 of 303 serious cases fall below the
triage cutoff, so no method can demonstrate a reliable gain there. The other
two measures are the `serious` label and the catch rate at a 5% false-alarm
rate. Whether the quiet cases are real or label noise is still an open
sponsor question; several of the six read as plain near misses.

**A. Features for hedged phrasing.** The production vectorizer removes
English stop words, which include "no", "not", "could", "would" and "may", so
"could have" and "no harm" never reach the model.

| Variant | Serious flagged | Labelled `serious` | Caught at 5% false alarms | Hurt recall | Unharmed flagged | Non-serious labelled `serious` |
|---|---|---|---|---|---|---|
| Production | 297/303 | 157/303 | 276/303 | 94.8% | 14.1% | 22 |
| Keep all stop words | 293 | 154 | 281 | 94.6% | 11.2% | 18 |
| Keep negations and modals | 295 | 160 | 280 | 95.0% | 12.5% | 19 |
| Hedge-phrase indicator features | 299 | 172 | 276 | 94.8% | 20.3% | 71 |

- No stop-word variant changes a serious-tier measure reliably (McNemar
  p >= 0.18 on every one).
- The hedge-phrase features label 15 more serious cases `serious` (p = 0.017,
  one of nine comparisons, so not robust), but triple the false `serious`
  labels (22 -> 71) and flag 20.3% of unharmed reports instead of 14.1%.
  Rejected.
- **Side finding, not adopted:** keeping negations and modals holds hurt
  recall (94.8% -> 95.0%) while flagging fewer unharmed reports
  (14.1% -> 12.5%). Each fold re-picks its cutoff, so this is the same recall
  target at lower cost. Worth a proper test as a production change, including
  October; nothing in `severity_model.py` was changed here.

**B. A second-stage check on reports below the cutoff.** A second stage always
flags more reports, so each is set against the production model with its
cutoff lowered to the same per-month share of unharmed reports flagged.

| Second stage | Unharmed flagged | Serious flagged | Lowered cutoff, same budget |
|---|---|---|---|
| None (production) | 14.1% | 297/303 | |
| Rule: high-alert medication mentioned | 25.1% | 298 | 300 |
| Rule: high-risk event mentioned | 18.7% | 299 | 299 |
| Rule: either | 29.2% | 300 | 302 |
| Learned serious-vs-rest detector | 22.0% | 302 | 299 |

- The word-list rules never beat simply lowering the cutoff, and catch fewer
  hurt cases overall at the same budget. Rejected. The lists were written
  after reading the six missed reports, so even these numbers flatter them.
- The learned detector (threshold picked on the cutoff month for 95% serious
  recall) recovers 5 of the 6 misses, 3 more than a lowered cutoff at the
  same budget (3 vs. 0 discordant cases, McNemar p = 0.25). Suggestive, not
  established, and the price is 3,808 more flagged reports across the eight
  months. It also catches fewer hurt cases overall than the lowered cutoff
  (6,087 vs. 6,136 of 6,366). Not adopted.

### Tested and rejected: grouping drug names (2026-10-13)

`drug_name_groups.py`, same rolling setup. Results in
`experiments/drug_name_groups_results.json`.

The intake fields hold only 47 distinct medication names, and 23.4% of
reports (26.8% of serious ones) mention one anywhere in the input text.
Three ways of telling the model which drugs belong together:

| Variant | Serious flagged | Labelled `serious` | Caught at 5% false alarms | Hurt recall | Unharmed flagged |
|---|---|---|---|---|---|
| Production | 297/303 | 157/303 | 276/303 | 94.8% | 14.1% |
| Add hand-mapped class tokens (17 classes + high-alert) | 297 | 155 | 276 | 94.9% | 14.3% |
| Replace drug names with class tokens | 297 | 159 | 278 | 94.8% | 14.2% |
| Add context-cluster tokens (k-means, k=8, train months only) | 297 | 155 | 277 | 94.9% | 14.3% |

No variant moves any serious-tier measure by more than 3 cases (McNemar
p >= 0.5 throughout), and no triage flag changed for any serious case.
Likely reasons: with 47 names the model already has enough examples to weight
each drug separately, and in this synthetic data harm rates barely differ by
drug (most sit within a few points of the training rate). The context
clusters also do not recover pharmacology (one cluster mixes heparin, insulin,
cephalosporins and venetoclax), which suggests the generator does not write
drug-specific narratives. This could differ on real reports, where the drug
vocabulary is far larger and misspelled.

---

## Event-type pipeline (teammate's `11-buckets` repo)

```mermaid
flowchart TD
    A["Report\n23 raw columns"] --> B["Same submission-time fields as\nseverity, plus: Event Type itself\nis blocked (it's the answer)"]
    B --> C["Glue into one string\n(identical format to severity side)"]
    C --> D["Text -> numbers: MiniLM\nsentence-embedding model\n-> 384 numbers (meaning, not word counts)"]
    D --> E["Head A: 11-way event type\n384 numbers -> 11 probabilities"]
    D --> F["Head B: hurt or not\n384 numbers + Head A's 11 probabilities\n-> 1 probability"]
    E --> F
    F --> G["Cutoff picked on validation\nfor >=95% recall, not a flat 50%"]
```

Code shape (`sentence-transformers`):

```python
SentenceTransformer("all-MiniLM-L6-v2")
model.encode(text)  # -> 384 numbers
```

Both heads trained together, 5 random seeds, one combined loss (event-type
loss + hurt loss).

### Results (teammate's README; independently spot-checked, see below)

| Metric | Value |
|---|---|
| Event-type accuracy (11-way) | 87.3% |
| Hurt recall (with event-type hint) | 95.7% ± 1.3% |
| Hurt precision (with hint) | 8.1% ± 0.3% |
| Hurt PR-AUC (with hint) | 0.472 ± 0.011 |
| Hurt PR-AUC (ablation, no hint) | 0.466 ± 0.005 |

Their own finding: the event-type hint barely moves PR-AUC (0.466 → 0.472).

**Independently verified:** rebuilt this exact architecture and ran it on our
own data as a sanity check (`combined_mtl.py`) — got 83.1% event-type
accuracy vs. their reported 87.3%. That's a 4-point gap, not a match — "same
ballpark, approach clearly works in principle" is the accurate read, not
"confirmed."

### Tested and rejected: merging the two pipelines

We tried feeding Head A's category hint into a severity prediction directly
(the same hint pattern, transplanted onto this repo's task). It made severity
worse, not better. **Rerun 2026-09-25 with fresh embeddings on the current
pipeline** (the previous number used an embedding cache older than the final
`data_pipeline.py`; deleted and regenerated to be sure): event-type accuracy
83.1% ± 0.6% (matches the earlier number, so the cache wasn't actually stale
in a way that mattered), severity macro F1 **0.451 ± 0.020 with the
event-type clue, 0.444 ± 0.005 without it** (3 seeds each) — confirms the
clue barely helps, within noise, same conclusion as before.

**The comparison that matters: 0.45 vs. the current production model's 0.82**
(Branch 2's derived 3-bucket macro F1 -- see the "Final, saved model" table
above), not the two-versions-ago 0.77 this section originally compared
against. Now genuinely apples-to-apples: same test set, same current
pipeline, both numbers real. The gap is enormous either way. Full results:
`combined_mtl_results.json`. The two pipelines stay separate because of this
result, not by default.

---

## Where the two pipelines meet

Never inside a model — only in the final combined record per report:

| | Severity (this repo) | Event type (teammate's repo) |
|---|---|---|
| Example report above | Grade E, treated/intervened (61% triage score) | Fall, hurt = yes (91%) |
| Feeds | Pre-fill + triage worklist | Recurring-theme tracking |

---

## The severity question is settled

Both pipelines produce a hurt/not-hurt signal (this repo's triage score;
the teammate's Head B, with the event-type hint). Same underlying question,
directly comparable:

| | This repo's model | Teammate's model (with hint) |
|---|---|---|
| Hurt PR-AUC | 0.871 | 0.472 |
| Precision at ~95% recall | 28.9% | 8.1% |

Not a close call. Use this repo's model for the harm-severity/triage
requirement; there's no case for reviving the merged approach (see above)
or switching architectures for this piece.

## One open question worth raising with the sponsor: Event Type

The sponsor's deck lists Event Type among the fields already provided to the
Safety Officer at intake. If reporters fill it in themselves when they file
the report, an 11-way classifier is predicting a field that already exists
in the data — not filling a gap. The sponsor's actual stated requirement for
this side of the project is multi-label **recurring themes** plus **trend
tracking**, which is a different task than single-label event-type
classification — it's closer to the clustering and theme-assignment work
already sketched (`semantic_clustering.py`). Worth confirming directly with
the sponsor whether Event Type is intake-filled before the teammate invests
further in the 11-way classifier head.
