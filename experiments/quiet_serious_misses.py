"""
Experiment: can the "quiet" serious cases be caught?

Some G/H/I reports score low because their narratives read like near misses
("could have received an incorrect dose", "no harm noted") rather than
describing an outcome. This tests two ideas from the README's to-do list,
all on the rolling evaluation (303 serious cases, see rolling_evaluation.py)
and paired against the production model on the same cases.

A. Features for hedged, potential-harm phrasing
   1. keep_stop_words        -- the production vectorizer removes English stop
                                words, which include "no", "not", "could",
                                "would" and "may", so "could have" and
                                "no harm" never reach the model. This keeps
                                every word.
   2. keep_negation_modals   -- removes stop words except negations and modals.
   3. hedge_lexicon          -- production TF-IDF plus hand-written indicator
                                features for hedged phrasing (HEDGE_PATTERNS),
                                and one for hedging next to a high-risk term.

B. A second-stage check on reports the triage cutoff does not flag
   4. rule: also flag a low-scoring report that mentions a high-alert
      medication or a high-risk event (HIGH_ALERT_MED / HIGH_RISK_EVENT).
   5. learned: also flag a low-scoring report that a separate
      serious-vs-everything-else classifier scores highly, with its threshold
      picked on the cutoff month only.

   A second stage always flags more reports, so each is set against the
   production model with its cutoff lowered to flag the same share of
   unharmed reports in each month. The question is whether the check beats
   simply lowering the cutoff.

Caveats:
- The word lists in B.4 were written after reading the six serious reports
  the production cutoff misses, so the rule's result on those six is
  optimistic. The learned check (B.5) has no such problem.
- Whether these quiet cases are real or label noise is an open sponsor
  question. A method that catches them is only useful if they are real.
- Development metric on synthetic data; October alone is the final test.

Run from the repo root:  .venv/bin/python -m experiments.quiet_serious_misses
Writes experiments/quiet_serious_misses_results.json (aggregates only).
"""

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from data_pipeline import load_dataset, rolling_folds
from rolling_evaluation import compare, compare_models, print_comparison, run_rolling
from severity_model import fit_model, make_vectorizer, pick_cutoff_for_recall

OUT_PATH = "experiments/quiet_serious_misses_results.json"
SERIOUS_SECOND_STAGE_RECALL = 0.95

KEPT_STOP_WORDS = {"no", "not", "nor", "never", "none", "nothing", "without", "cannot",
                   "could", "would", "should", "might", "may", "must", "almost"}

HEDGE_PATTERNS = {
    "modal_have": r"\b(?:could|would|might|may|should) have\b",
    "potential": r"\bpotential(?:ly)?\b",
    "near_miss": r"\bnear[- ]miss(?:es)?\b",
    "almost": r"\b(?:almost|nearly|about to)\b",
    "caught_before": r"\b(?:caught|identified|noticed|discovered|intercepted|stopped) (?:before|prior to)\b",
    "not_given": r"\b(?:not|never) (?:given|administered)\b",
    "no_harm": r"\bno (?:harm|injury|adverse)\b",
    "risk": r"\b(?:risk|unsafe|danger\w*)\b",
    "gravity": r"\b(?:serious|severe|significant|critical)\b",
}

# ISMP-style high-alert medication classes, plus the drugs in this dataset
# that belong to them.
HIGH_ALERT_MED = (
    r"\b(?:anticoagul\w*|heparin|warfarin|enoxaparin|apixaban|insulin|opioid\w*|narcotic\w*|morphine|"
    r"hydromorphone|oxycodone|fentanyl|chemo\w*|venetoclax|methotrexate|epinephrine|vasopressor\w*|"
    r"milrinone|potassium|sedat\w*|paralytic\w*|tacrolimus|digoxin|high[- ]alert)\b"
)
HIGH_RISK_EVENT = (
    r"\b(?:died|death|deceased|expired|cardiac arrest|code blue|rapid response|unresponsive|intubat\w*|"
    r"resuscitat\w*|transfus\w*|blood product\w*|wrong patient|wrong[- ]site|wrong[- ]side|"
    r"transferred to (?:the )?icu|icu transfer|permanent\w*|hemorrhag\w*|anaphyla\w*|"
    r"respiratory (?:arrest|failure|depression)|overdose\w*|sepsis|stroke|fracture\w*)\b"
)
HIGH_RISK = f"{HIGH_ALERT_MED}|{HIGH_RISK_EVENT}"


class HedgeLexicon(BaseEstimator, TransformerMixin):
    """One 0/1 column per HEDGE_PATTERNS entry, plus one for any hedge
    appearing in a report that also has a high-risk term."""

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        low = pd.Series(texts).str.lower()
        cols = [low.str.contains(p, regex=True).to_numpy() for p in HEDGE_PATTERNS.values()]
        any_hedge = np.any(cols[:6], axis=0)  # the phrasing patterns, not no_harm/risk/gravity
        cols.append(any_hedge & low.str.contains(HIGH_RISK, regex=True).to_numpy())
        return sparse.csr_matrix(np.column_stack(cols).astype(float))


def stop_word_variant(stop_words):
    # Same settings as make_vectorizer(), different stop list.
    return TfidfVectorizer(stop_words=stop_words, max_features=20000, ngram_range=(1, 2), min_df=3,
                           token_pattern=r"(?u)\b[\w-]+\b")


FEATURE_VARIANTS = {
    "keep_stop_words": lambda: stop_word_variant(None),
    "keep_negation_modals": lambda: stop_word_variant(sorted(ENGLISH_STOP_WORDS - KEPT_STOP_WORDS)),
    "hedge_lexicon": lambda: FeatureUnion([("tfidf", make_vectorizer()), ("hedge", HedgeLexicon())]),
}


def learned_second_stage(df):
    """Per fold, a serious-vs-everything-else classifier on the production
    features. Returns a boolean Series indexed by event_no: True where the
    score clears the threshold picked on that fold's cutoff month."""
    flags = []
    for train_df, cutoff_df, test_df in rolling_folds(df):
        vectorizer = make_vectorizer()
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(vectorizer.fit_transform(train_df["text"]), train_df["bucket"] == "serious")
        cutoff_scores = clf.predict_proba(vectorizer.transform(cutoff_df["text"]))[:, 1]
        threshold = pick_cutoff_for_recall((cutoff_df["bucket"] == "serious").to_numpy(), cutoff_scores,
                                           SERIOUS_SECOND_STAGE_RECALL)
        test_scores = clf.predict_proba(vectorizer.transform(test_df["text"]))[:, 1]
        flags.append(pd.Series(test_scores >= threshold, index=test_df["event_no"]))
    return pd.concat(flags)


def second_stage_result(oof, extra_flag):
    """Compare production + second stage against (a) production alone and
    (b) production with its cutoff lowered to the same per-month share of
    unharmed reports flagged."""
    with_check = oof["flagged"] | extra_flag
    none = oof["bucket"] == "none"
    budget = with_check[none].groupby(oof.loc[none, "event_month"]).mean()
    lowered_cutoff = oof["none_pct"] >= 1 - oof["event_month"].map(budget)

    def count(flag, mask):
        return f"{int(flag[mask].sum())}/{int(mask.sum())}"

    serious, hurt = oof["bucket"] == "serious", oof["hurt_label"] == 1
    natural_serious = serious & (oof["split"] != "TRAIN")
    return {
        "reports_added_by_check": int((with_check & ~oof["flagged"]).sum()),
        "unharmed_flagged": {"production": float(oof.loc[none, "flagged"].mean()),
                             "with_check": float(with_check[none].mean())},
        "serious_flagged": {"production": count(oof["flagged"], serious),
                            "with_check": count(with_check, serious),
                            "lowered_cutoff_same_budget": count(lowered_cutoff, serious)},
        "serious_flagged_natural_rate_months": {
            "production": count(oof["flagged"], natural_serious),
            "with_check": count(with_check, natural_serious),
            "lowered_cutoff_same_budget": count(lowered_cutoff, natural_serious)},
        "hurt_flagged": {"production": count(oof["flagged"], hurt),
                         "with_check": count(with_check, hurt),
                         "lowered_cutoff_same_budget": count(lowered_cutoff, hurt)},
        "check_vs_lowered_cutoff_on_serious": compare(oof.assign(x=lowered_cutoff), oof.assign(x=with_check), column="x"),
    }


def main():
    df = load_dataset()
    print("Baseline (production model):")
    baseline, _ = run_rolling(df=df)

    results = {"feature_variants": {}, "second_stage": {}}
    for name, make in FEATURE_VARIANTS.items():
        print(f"\nVariant {name}:")
        variant, _ = run_rolling(fit_fn=lambda train_df: fit_model(train_df, vectorizer=make()), df=df, verbose=False)
        results["feature_variants"][name] = compare_models(baseline, variant)
        print_comparison(name, results["feature_variants"][name])

    low = baseline["text"].str.lower()
    below = ~baseline["flagged"]
    checks = {
        "rule_high_alert_medication": below & low.str.contains(HIGH_ALERT_MED, regex=True),
        "rule_high_risk_event": below & low.str.contains(HIGH_RISK_EVENT, regex=True),
        "rule_either": below & low.str.contains(HIGH_RISK, regex=True),
        "learned_serious_detector": below & baseline["event_no"].map(learned_second_stage(df)),
    }
    print("\nSecond-stage checks (lowered cutoff = production cutoff lowered to the same unharmed flag rate):")
    for name, extra in checks.items():
        r = second_stage_result(baseline, extra)
        results["second_stage"][name] = r
        print(f"\n--- {name}: adds {r['reports_added_by_check']} flagged reports ---")
        print(f"  unharmed flagged: {r['unharmed_flagged']['production']:.1%} -> {r['unharmed_flagged']['with_check']:.1%}")
        for key in ("serious_flagged", "serious_flagged_natural_rate_months", "hurt_flagged"):
            v = r[key]
            print(f"  {key:38s} production {v['production']}, with check {v['with_check']}, "
                  f"lowered cutoff {v['lowered_cutoff_same_budget']}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
