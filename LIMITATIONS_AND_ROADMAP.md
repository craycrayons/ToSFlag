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
judgment. It now scores a user's own pre-split clauses; it cannot yet take a raw
contract as a wall of text, because clause segmentation is unbuilt (see roadmap).

---

## Honest limitations (the present situation)

1. **Segmentation, not scoring, is now the gap to arbitrary text.** The
   inference path is built: `scripts/infer.py` scores an already-split clause
   list (one per line) with a frozen model, no training and no network, and
   `scripts/train_and_save.py` freezes the model + operating point together. So
   a user *can* now feed in their own clauses and get a report. What they cannot
   yet do is paste a raw ToS as a wall of text - that requires clause
   segmentation, which is not built (see roadmap). The scoring gap is closed;
   the segmentation gap is the remaining barrier between "score my clauses" and
   "score my contract".

2. **Single-annotator community dataset.** `CodeHima/TOS_Dataset` (MIT licensed)
   is conveniently shaped but carries one annotator's judgement and is not the
   peer-reviewed CLAUDETTE release. It even contains directional noise: at least
   one consumer-protective carve-out is labelled unfair. The LexGLUE `unfair_tos`
   cross-check (now run) is the validation that substantiates claims beyond this
   one dataset; the model holds 0.895 recall there as held-out data.

3. **TF-IDF is shallow.** Lap 1 keys on surface phrasing ("sole discretion", "we
   reserve the right"). Adversarial rephrasing would evade it. The error analysis
   predicted the residual misses were mostly semantic - plus a hard tail where the
   trigger word is present but buried in syntax the model cannot parse (a
   class-action waiver phrased as mutual, an arbitration carve-out worded to sound
   even-handed). Lap 2 tested this: legal-BERT recovers ~a third of those misses
   (confirming part of the diagnosis) but trades recall for precision rather than
   strictly improving, and a hard residual resists both models - so the ceiling is
   clause-level context and label consistency, not model depth. The shipped
   headline model is still the shallow one, by choice (CPU, reproducible).

4. **English only, clause level.** No multilingual support, and no document-level
   reasoning or clause-to-clause interaction is modelled.

---

## Roadmap (where it is going, in order)

### Next: raw-document segmentation (the inference path is now shipped)

The inference path - the small piece - is built and committed.

- **The small piece - the inference path (shipped).** `scripts/train_and_save.py`
  fits the Lap-1 pipeline on cleaned data, derives the committed recall-first
  threshold on validation, and freezes both to `models/` as one unit (the model
  and its operating point cannot drift apart). `scripts/infer.py` loads that
  frozen model, scores an already-split clause list (one per line, from a file or
  piped stdin), applies the committed threshold, and writes the clause report
  minus the ground-truth columns (`true_label`, `outcome`, `severity`) because
  unseen text has no answer key. Output: `clause`, `flagged`, `risk_score`,
  `why_flagged`, as CSV and colour-coded XLSX. It reuses the existing
  `build_pipeline` and the exact `why_flagged` logic by import, not
  reimplementation, so it cannot diverge from the trained artifact.

- **The hard piece - clause segmentation (not built).** The dataset hands clauses
  pre-split, one per row. A real ToS is a wall of text. Cutting it into clauses is
  a genuine problem: naive splitting on periods breaks on "Inc.", "e.g.", section
  numbers, and nested sub-clauses; splitting on newlines depends on how the text
  was copy-pasted. It will not be perfect, and the segmentation quality bounds the
  whole tool's usefulness more than the model does. This is named here rather than
  quietly shipped because a half-working "paste your ToS" feature that segments
  badly is worse than not having one. The shipped inference path sidesteps it by
  accepting a pre-split list - about 80% of the utility for 20% of the work.
  Raw-document segmentation is the follow-on.

### Then: the LLM depth stage (two-stage architecture)

The shipped classifier is the *breadth* stage - wide, cheap, reproducible,
deterministic. The natural follow-on is a *depth* stage: an LLM judge applied
only to the small subset the classifier surfaces, giving explanation and
catching the cumulative-scope cases a bag-of-words model structurally cannot see
(a licence unfair by "perpetual, irrevocable, worldwide" pileup, no single scary
token). The design decision is the triage rule, and it is deliberately *not*
"send the extremes" - the extremes are where the cheap model is already right.
Route the union of two cheap signals (score near the threshold; a
length/complexity flag or a Lap-1-vs-Lap-2 disagreement) to the LLM, and keep an
agreement column so a human sorts by where the two stages disagreed. The
principled version derives the uncertain band by measuring per-score-bin
classifier reliability on a holdout rather than guessing it. This is the
composition - ML + deterministic logic + system design - that the project is
meant to demonstrate.

### Then: validation and ranking quality

- **LexGLUE `unfair_tos` cross-check (done).** Trains on the community set and
  evaluates on the peer-reviewed benchmark collapsed to binary. It quantified
  this dataset's labelling boundary against expert labels and substantiated
  generalisation: 0.895 recall on the held-out expert set, with the recall-first
  threshold transferring almost unchanged. This was the single most valuable
  unrun validation; it is now run (`reports/crosscheck.md`).

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
| Recall-first cost-asymmetry threshold | Shipped, validated (0.685 recall, cleaned) |
| Readable per-clause report (CSV + XLSX) | Shipped |
| Error analysis of the model's own misses | Shipped |
| Structural non-clause cleaning (~18% of rows) | Shipped (`drop_nonclauses`) |
| Legal-BERT semantic-recovery comparison | Shipped (32% recovery; trades recall for precision) |
| Score a user's own pasted clauses (pre-split) | Shipped (`infer.py` + `train_and_save.py`) |
| Clause segmentation of raw documents | Not built (the hard part) |
| Two-stage classifier + LLM depth pass | Not built (next architecture lap) |
| LexGLUE cross-check | Shipped (0.895 recall held-out) |
| Calibrated severity ranking | Not built |
| Deployed service (API + Docker) | Not built (Lap 3) |
