# Experiments

Retired and one-off scripts, kept for history. None of them are part of the
pipeline (`data_pipeline.py`, `severity_model.py`,
`severity_model_extended.py` in the repo root). Results and conclusions are
written up in [PIPELINE.md](../PIPELINE.md).

| Script | What it tested | Outcome |
|---|---|---|
| `baseline_grader.py` | TF-IDF + LogReg trained directly on the 3 buckets | Superseded by the 10-grade model |
| `ordinal_vs_coarse.py` | 10-grade model vs. a model trained only on hurt / not hurt | 10-grade model kept; became `severity_model.py` |
| `combined_mtl.py` | Feeding a predicted event type into severity (frozen MiniLM) | Rejected |
| `augment_rare_grades.py` | Hand-written extra training reports for grades G/H/I | Rejected, no measurable effect |
| `finetune_distilbert.py` | DistilBERT fine-tuned on the 3 buckets | Macro F1 0.79 vs. production's 0.82; still a candidate |
| `leakage_inflation_check.py` | Effect of adding the two post-review fields back into the input | Little effect on this synthetic data |
| `longformer_colab.ipynb` | Clinical-Longformer on the 3 buckets, for Colab | Stale: predates the leakage fix and date split |
| `quiet_serious_misses.py` | Hedged-phrasing features and a second-stage check, to catch serious cases that read like near misses | Rejected; side finding on stop words still open |
| `drug_name_groups.py` | Grouping the 47 drug names by hand-mapped class or by context clusters | Rejected, no measurable effect |

The last two run on the rolling evaluation (`rolling_evaluation.py`) and write
their aggregate results next to themselves as `*_results.json`.

Run them as modules from the repo root, so they can import `data_pipeline`
and so their relative output paths resolve there:

```bash
python3 -m experiments.baseline_grader
```

`finetune_distilbert.py` and `combined_mtl.py` also need torch and the
transformer packages. `leakage_inflation_check.py` reads the
excluded columns on purpose, to measure their effect.
