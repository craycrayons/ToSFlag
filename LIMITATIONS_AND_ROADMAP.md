# ToSFlag - Limitations and Roadmap

What this project is honest about, and where it is going. This document exists
because naming the present situation precisely - what works, what does not, and
why the gaps are hard - is part of the work, not an apology for it.

---

## What ToSFlag is, exactly

A recall-first classifier that scores Terms-of-Service clauses for likely
unfairness to a consumer, tuned on an explicit cost asymmetry (a missed unfair
clause costs more than a needlessly-reviewed fair one) rather than on accuracy
or F1. It ships with a readable per-clause report, an error analysis of its own
misses, and an apples-to-apples Lap 2 comparison against a legal-domain encoder.

It is a validated scoring engine and a portfolio-grade demonstration of metric
judgment. It is not yet a tool a stranger can point at their own contract.

---

## Honest limitations (the present situation)

1. **It runs on one dataset, not on arbitrary text.** The repo trains on
   `CodeHima/TOS_Dataset` and reports on that dataset's own held-out test split.
   Every clause in the output was already in the dataset and already labelled.
   There is no path today for a user to feed in their own ToS and get a report.
   The model can score new text; nothing wired up exposes that to a user. This
   is the single biggest gap between "artifact" and "tool", and it is the
   headline next step below.

2. **Single-annotator community dataset.** `CodeHima/TOS_Dataset` (MIT licensed)
   is conveniently shaped but carries one annotator's judgement and is not the
   peer-reviewed CLAUDETTE release. It even contains directional noise: at least
   one consumer-protective carve-out is labelled unfair. The planned LexGLUE
   `unfair_tos` cross-check is the validation that would substantiate any claim
   beyond this one dataset.

3. **TF-IDF is shallow.** Lap 1 keys on surface phrasing ("sole discretion", "we
   reserve the right"). Adversarial rephrasing would evade it. The error analysis
   already showed the residual misses are semantic, not lexical, which is the
   evidenced case for the legal-BERT lap - but the shipped headline model is
   still the shallow one.

4. **English only, clause level.** No multilingual support, and no document-level
   reasoning or clause-to-clause interaction is modelled.

---

## Roadmap (where it is going, in order)

### Next: bring-your-own-ToS inference

This is the gap that turns the artifact into a tool. It is two pieces of work,
and only one of them is small.

- **The small piece - the inference path.** Load a cached trained model, run the
  existing pipeline's scoring on new unlabelled text, apply the existing
  threshold, write the same clause report minus the ground-truth columns
  (true_label, outcome, severity) because there is no answer key for text the
  model has never seen. The output for a user's own document is: clause, flagged,
  risk_score, why_flagged. Roughly 60 lines reusing what already exists.

- **The hard piece - clause segmentation.** The dataset hands clauses pre-split,
  one per row. A real ToS is a wall of text. Cutting it into clauses is a genuine
  problem: naive splitting on periods breaks on "Inc.", "e.g.", section numbers,
  and nested sub-clauses; splitting on newlines depends on how the text was
  copy-pasted. It will not be perfect, and the segmentation quality bounds the
  whole tool's usefulness more than the model does. This is named here rather
  than quietly shipped because a half-working "paste your ToS" feature that
  segments badly is worse than not having one.

  A first version sidesteps this by accepting an already-split clause list (one
  per line) - about 80% of the utility for 20% of the work. Raw-document
  segmentation is the harder follow-on.

### Then: validation and ranking quality

- **LexGLUE `unfair_tos` cross-check.** Train on the community set, evaluate on
  the peer-reviewed benchmark collapsed to binary. Quantifies this dataset's
  label noise and substantiates generalisation beyond it. The single most
  valuable unrun validation.

- **Calibrated probabilities (temperature scaling)** for the severity ranking, so
  flagged clauses can be ordered by a trustworthy confidence rather than a raw
  score.

### Ceiling: deployment

- **FastAPI service + Docker + CI**, a clause-level review queue ranked by
  calibrated severity. This is the step that makes it a deployable service rather
  than a script. Lap 3 scope; not needed for a portfolio artifact.

---

## What is shipped vs promised (so the line is clear)

| Capability | Status |
|---|---|
| Recall-first cost-asymmetry threshold | Shipped, validated (0.796 recall) |
| Readable per-clause report (CSV + XLSX) | Shipped |
| Error analysis of the model's own misses | Shipped |
| Legal-BERT semantic-recovery comparison | Shipped (rerun pending for the corrected number) |
| Score a user's own pasted ToS | Not built (next step) |
| Clause segmentation of raw documents | Not built (the hard part) |
| LexGLUE cross-check | Not run |
| Calibrated severity ranking | Not built |
| Deployed service (API + Docker) | Not built (Lap 3) |
