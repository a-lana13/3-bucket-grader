"""
Experiment: does grouping drug names help the severity model?

Medication names reach the model as separate words ("oxyCODONE",
"HYDROmorphone"), so it has to learn each drug's weight on its own. This
tests whether telling it which drugs belong together helps, on the rolling
evaluation (303 serious cases, see rolling_evaluation.py), paired against
the production model on the same cases.

The dataset has 47 distinct medication names across the three intake fields.

1. class_tokens     -- for each drug mentioned anywhere in the input text, add
                       a token for its pharmacological class (DRUG_CLASSES,
                       hand-mapped) and one for ISMP-style high-alert drugs.
2. class_replace    -- the same class tokens, but the drug name itself is
                       removed, so the model sees only the group.
3. context_clusters -- no hand mapping. In each fold, on training months
                       only, each drug is described by the average TF-IDF
                       vector of the narratives that mention it (drug names
                       masked out), the drugs are grouped with k-means, and a
                       cluster token is added.

The class map and cluster tokens are built from the input text alone (the
intake prefix plus narrative), so nothing here reads a post-review field.

Development metric on synthetic data; October alone is the final test.

Run from the repo root:  .venv/bin/python -m experiments.drug_name_groups
Writes experiments/drug_name_groups_results.json (aggregates only).
"""

import json
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from data_pipeline import load_dataset
from rolling_evaluation import compare_models, print_comparison, run_rolling
from severity_model import fit_model, make_vectorizer

OUT_PATH = "experiments/drug_name_groups_results.json"
N_CLUSTERS = 8
MIN_MENTIONS = 20  # a drug needs this many training reports to be clustered

# All 47 medication names in the dataset's intake fields, lowercased.
DRUG_CLASSES = {
    "opioid": ["hydromorphone", "fentanyl", "morphine", "oxycodone"],
    "anticoagulant": ["apixaban", "enoxaparin", "heparin", "warfarin"],
    "insulin": ["insulin lispro", "insulin regular"],
    "antibiotic": ["amoxicillin-clavulanate", "ampicillin", "cefazolin", "ceftriaxone", "cefepime",
                   "ciprofloxacin", "clindamycin", "doxycycline", "levofloxacin", "meropenem",
                   "metronidazole", "piperacillin-tazobactam", "sulfamethoxazole-trimethoprim", "vancomycin"],
    "analgesic_nonopioid": ["acetaminophen", "ibuprofen"],
    "antiplatelet": ["aspirin"],
    "vasoactive": ["epinephrine", "milrinone"],
    "diuretic": ["bumetanide", "furosemide"],
    "electrolyte": ["calcium gluconate", "potassium phosphate"],
    "antihypertensive": ["clonidine", "metoprolol"],
    "corticosteroid": ["dexamethasone", "budesonide-formoterol"],
    "antihistamine_acid": ["diphenhydramine", "famotidine"],
    "immunomodulator": ["secukinumab", "tocilizumab", "tacrolimus", "pegfilgrastim"],
    "chemotherapy": ["venetoclax"],
    "contrast": ["iohexol"],
    "iv_fluid": ["lactated ringers injection"],
    "unknown": ["unknown medication"],
}
HIGH_ALERT_CLASSES = {"opioid", "anticoagulant", "insulin", "vasoactive", "chemotherapy"}
HIGH_ALERT_DRUGS = {"potassium phosphate", "tacrolimus"}

DRUG_TO_CLASS = {drug: cls for cls, drugs in DRUG_CLASSES.items() for drug in drugs}
# Longest first, so "insulin lispro" matches before a shorter name could.
DRUG_PATTERN = re.compile(r"\b(" + "|".join(re.escape(d) for d in sorted(DRUG_TO_CLASS, key=len, reverse=True)) + r")\b")


def drugs_in(text):
    return DRUG_PATTERN.findall(text.lower())


def class_tokens(drugs):
    tokens = {f"drugclass_{DRUG_TO_CLASS[d]}" for d in drugs}
    if any(DRUG_TO_CLASS[d] in HIGH_ALERT_CLASSES or d in HIGH_ALERT_DRUGS for d in drugs):
        tokens.add("drug_highalert")
    return sorted(tokens)


class AugmentedVectorizer:
    """The production vectorizer over text rewritten by `augment`."""

    def __init__(self, augment):
        self.augment = augment
        self.vectorizer = make_vectorizer()

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(self.augment(pd.Series(texts)))

    def transform(self, texts):
        return self.vectorizer.transform(self.augment(pd.Series(texts)))


def add_class_tokens(texts):
    return texts.map(lambda t: t + "\n" + " ".join(class_tokens(drugs_in(t))))


def replace_with_class_tokens(texts):
    return texts.map(lambda t: DRUG_PATTERN.sub(" ", t.lower()) + "\n" + " ".join(class_tokens(drugs_in(t))))


class ContextClusterVectorizer(AugmentedVectorizer):
    """Groups drugs by the narratives they appear in, using the fit data only."""

    def __init__(self):
        super().__init__(self.add_cluster_tokens)
        self.drug_to_cluster = {}

    def add_cluster_tokens(self, texts):
        def tokens(t):
            return sorted({f"drugcluster_{self.drug_to_cluster[d]}" for d in drugs_in(t) if d in self.drug_to_cluster})
        return texts.map(lambda t: t + "\n" + " ".join(tokens(t)))

    def fit_transform(self, texts):
        texts = pd.Series(texts).reset_index(drop=True)
        mentions = texts.map(drugs_in)
        # Describe each drug by its reports' text with every drug name masked,
        # so drugs group by context rather than by their own names.
        context = make_vectorizer().fit_transform(texts.map(lambda t: DRUG_PATTERN.sub(" ", t.lower())))
        drugs, profiles = [], []
        for drug in DRUG_TO_CLASS:
            rows = np.flatnonzero(mentions.map(lambda found: drug in found).to_numpy())
            if len(rows) >= MIN_MENTIONS:
                drugs.append(drug)
                profiles.append(np.asarray(context[rows].mean(axis=0)).ravel())
        labels = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0).fit_predict(normalize(np.vstack(profiles)))
        self.drug_to_cluster = dict(zip(drugs, labels.tolist()))
        return super().fit_transform(texts)


VARIANTS = {
    "class_tokens": lambda: AugmentedVectorizer(add_class_tokens),
    "class_replace": lambda: AugmentedVectorizer(replace_with_class_tokens),
    "context_clusters": ContextClusterVectorizer,
}


def main():
    df = load_dataset()
    mentions = df["text"].map(drugs_in)
    coverage = {
        "reports_mentioning_a_listed_drug": float((mentions.map(len) > 0).mean()),
        "serious_reports_mentioning_a_listed_drug": float((mentions[df["bucket"] == "serious"].map(len) > 0).mean()),
    }
    print(f"Reports mentioning a listed drug: {coverage['reports_mentioning_a_listed_drug']:.1%} "
          f"(serious reports: {coverage['serious_reports_mentioning_a_listed_drug']:.1%})")

    print("\nBaseline (production model):")
    baseline, _ = run_rolling(df=df)

    results = {"coverage": coverage, "variants": {}}
    last_clusters = {}
    for name, make in VARIANTS.items():
        print(f"\nVariant {name}:")

        def fit(train_df):
            vectorizer, clf = fit_model(train_df, vectorizer=make())
            if name == "context_clusters":
                last_clusters.update(vectorizer.drug_to_cluster)
            return vectorizer, clf

        variant, _ = run_rolling(fit_fn=fit, df=df, verbose=False)
        results["variants"][name] = compare_models(baseline, variant)
        print_comparison(name, results["variants"][name])

    # The clusters from the last fold (the production training set), to show
    # what the context grouping actually found.
    clusters = {}
    for drug, c in sorted(last_clusters.items(), key=lambda kv: kv[1]):
        clusters.setdefault(f"cluster_{c}", []).append(f"{drug} ({DRUG_TO_CLASS[drug]})")
    results["context_clusters_last_fold"] = clusters
    print("\nContext clusters in the last fold:")
    for c, members in clusters.items():
        print(f"  {c}: {', '.join(members)}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
