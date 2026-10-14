# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Capstone project: grades free-text patient safety incident reports on the 10-grade PSRS harm scale (A, B1, B2, C, D, E, F, G, H, I) for triage. One TF-IDF + logistic regression model produces a letter grade, a derived 3-bucket label (`none`/`some`/`serious`), and a calibrated triage score. The repo name is historical: the 3-bucket model was the first version and is now derived from the 10-grade model.

A teammate's separate `11-buckets` repo handles event-type classification. The two pipelines are deliberately kept separate (merging was tested and rejected, see `experiments/combined_mtl.py`); `data_pipeline.py` mirrors that repo's `src/pipeline.py` constants (prefix fields, excluded columns, event buckets) so both evaluate on the same rows.

`README.md` is the current summary. `PIPELINE.md` is a long dated changelog of experiments and review rounds; earlier entries contain claims that later entries correct inline (`(corrected in round N)`), so read the whole thread before quoting a number or claim from it.

## Commands

Flat directory of standalone scripts. No package, build, or linter.

```bash
.venv/bin/python data_pipeline.py             # sanity check: split sizes and bucket distribution
.venv/bin/python severity_model.py            # train + evaluate + save production model to severity_model/
.venv/bin/python severity_model_extended.py   # extended evaluation; needs the saved model from the step above
.venv/bin/python prefill_extra_fields.py      # HPI Designation classifier; reuses the saved vectorizer
.venv/bin/python error_analysis.py            # harmed test reports the triage cutoff misses; needs the saved model
.venv/bin/python rolling_evaluation.py        # retrain per month, score G/H/I on 303 cases (a few minutes); retrains, no saved model needed
.venv/bin/python predict.py reports.csv -o out.csv   # score a CSV/parquet of raw reports with the saved model

.venv/bin/python -m pytest                                        # all tests (~2s, no dataset needed)
.venv/bin/python -m pytest tests/test_data_pipeline.py -k split   # one file / tests matching a name
```

- Tests live in `tests/` and cover the leakage guard, split decoding, grade → bucket mapping, probability sums, and a round trip of the saved model through `predict.py`. `tests/test_rolling.py` covers the rolling folds and the paired comparison. `tests/test_saved_model.py` is skipped when `severity_model/*.joblib` is absent. `pytest.ini` puts the repo root on the import path.
- Run scripts from the repo root: scripts import `data_pipeline` as a sibling module and write to relative paths (`./severity_model`, `./distilbert_output`, `./_mtl_embed_*.npy`). Scripts in `experiments/` must be run as modules for the same reason: `.venv/bin/python -m experiments.baseline_grader`, not `python experiments/baseline_grader.py`.
- `.venv` is a uv-managed Python 3.11 env with only pandas, pyarrow, scikit-learn, joblib, and pytest. There is no requirements file; add packages with `uv pip install`. The transformer experiments (`finetune_distilbert.py` and `combined_mtl.py` in `experiments/`) additionally need torch, transformers, datasets, accelerate, and sentence-transformers, which are not installed.
- The dataset is not in the repo. `DATA_PATH` at the top of `data_pipeline.py` is a hardcoded absolute path to `LOCAL_ONLY_student_facing_candidate.parquet` (currently `/home/alana/capstone/`, one level above the repo). It differs per machine, so a `DATA_PATH` edit shows up as an uncommitted change.
- `*.parquet`, `*.csv`, `LOCAL_ONLY*`, and `severity_model/*.joblib` are gitignored. Never commit the data or anything derived row-by-row from it.

## Architecture

### `data_pipeline.py` is the single source of truth

Every script gets its data through `train_val_test()`, and imports label constants from here rather than redefining them. It:

- **Enforces leakage safety.** Model input is `event_comments` plus a labeled prefix of intake-time fields (`PREFIX_FIELDS`: unit, service, age, medication names/doses), assembled by `build_input_text()`. `EXCLUDED_COLUMNS` (manager comments, unit actions, shareable lessons, HPI Designation, Analyst-Report Type, Level of Investigation) are filled in after review and must never reach model features; `assert_no_excluded_columns()` raises `LeakageError` if they do. Inference must go through `build_input_text()` too, so train and predict text match.
- **Splits by date, not randomly.** The split is decoded from the `Event No.` prefix (`SYNPROD-TRAIN-…` / `VALIDATION` / `TEST` = Jan–Aug / Sep / Oct). Train deliberately oversamples harm (~15%) versus the natural rate in val/test (~5%).
- **Derives all label views from the one harm-score column:** `full_label` (10 grades, ordinal index into `FULL_LABELS`), `bucket`/`label` (3-way via `BUCKET_MAP`), `hurt_label` (binary, grades E–I), plus auxiliary `event_type_label`, `hpi_label`, `report_type_label` (`UNLABELED = -1` where missing) and subgroup columns used only for evaluation breakdowns.
- Raises `DataContractError` on unexpected harm scores, event codes, or `Event No.` formats instead of silently dropping rows. `Deleted` harm scores (expected in the sponsor's real data, absent in the synthetic set) are dropped with a printed count.

### Production path

`severity_model.py` trains one multinomial LogReg on the 10 grades and derives everything else from its probabilities:

- 3-bucket label = argmax over probabilities summed per `BUCKET_GROUPS` (`bucket_probs`)
- triage score = summed probability of grades E–I (`triage_score`)
- triage cutoff = highest cutoff reaching ≥95% recall on **validation**, then frozen and applied to test
- Platt-scaling calibrator fit on validation, to correct for train's inflated harm rate

It writes `vectorizer.joblib`, `model.joblib`, `calibrator.joblib`, `config.json` (frozen cutoff), and `evaluation.json` to `severity_model/`, then reloads them in `demo_predict()` as a round-trip check.

`predict.py` is the inference entry point: it scores a CSV/parquet of raw reports with the saved artifacts, importing `bucket_probs`/`triage_score` from `severity_model.py`. Rows with no input text are left unscored, matching the training-time drop.

`rolling_evaluation.py` is the development metric for the serious tier. It walks `rolling_folds()` from `data_pipeline.py` (train on months up to k-2, cutoff on month k-1, test on month k; test months Mar–Oct), fitting each fold with `fit_model()` from `severity_model.py`, and writes `severity_model/rolling_evaluation.json`. The last fold is the production split and must reproduce the saved cutoff. To measure a change, pass a different fit function to `run_rolling()` and pair the two results with `compare()`. Only pool rates that don't depend on the harm rate (recall, false-alarm rate); precision and flag rate differ between the oversampled Jan–Aug months and Sep/Oct.

`severity_model_extended.py`, `prefill_extra_fields.py`, `error_analysis.py` and `predict.py` load those saved artifacts rather than retraining, so they must run after `severity_model.py`. `severity_model_extended.py` carries its own copies of `triage_score`/`bucket_probs`; keep them in sync with `severity_model.py` if either changes.

### Everything else is a retired or one-off experiment

Everything in `experiments/` (`baseline_grader.py`, `ordinal_vs_coarse.py`, `finetune_distilbert.py`, `combined_mtl.py`, `augment_rare_grades.py`, `leakage_inflation_check.py`, `longformer_colab.ipynb`, `quiet_serious_misses.py`, `drug_name_groups.py`) is kept for history and documented in `PIPELINE.md` and `experiments/README.md`. None of it is part of the pipeline. `PIPELINE.md` and code comments refer to these files by bare filename. `combined_mtl_results.json` stays in the repo root, where `combined_mtl.py` writes it. Notes:

- `leakage_inflation_check.py` reads the excluded columns on purpose, to measure their effect.
- `longformer_colab.ipynb` predates the leakage fix and date split (random split, 3-bucket only); its comparison table is stale.

## Conventions that matter here

- **Tune on validation only.** Any cutoff, threshold, or calibration is fit on September and reported on October. A number fit on test must be labeled as optimistic (PIPELINE.md rounds 5–6 document this mistake being made and corrected).
- **Committed JSON is the canonical record.** Results quoted in `README.md` and `PIPELINE.md` are read from `severity_model/evaluation.json` and `extended_evaluation.json`. Rerunning `severity_model.py` overwrites `evaluation.json` and `config.json`, and results shift slightly between environments (e.g. cutoff 0.2361 vs 0.2362). Don't commit regenerated JSON incidentally; if it is committed, rerun the extended evaluation and update the docs' numbers to match.
- **Keep the TF-IDF `token_pattern=r"(?u)\b[\w-]+\b"`** in anything that mirrors the production vectorizer. sklearn's default drops single characters, turning "X-ray" into "ray".
- **The serious tier is tiny** (20 G/H/I test cases, 16 validation, zero grade-I test cases). Report counts and confidence intervals alongside any rate on it, and on subgroups. Use `rolling_evaluation.py` (303 cases) to compare models on it, but October stays the final test: Jan–Aug informed every modelling choice.
- **The 95% recall target is this project's own default, not a sponsor requirement.** The sponsor comparison (95.2% sensitivity at 96.0% specificity) is not like-for-like: their script uses the excluded fields, a random split, and real data.
- **Data is synthetic.** Don't present findings as holding on real reports. Open sponsor questions are tracked at the end of the README.
- Experiments that don't pan out are documented in `PIPELINE.md` as tested-and-rejected rather than deleted.
