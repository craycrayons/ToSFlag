# ToSFlag

A recall-first detector for potentially unfair clauses in online Terms of Service.

Most clause-classification projects optimise accuracy or F1 and report a single
number. ToSFlag is built on a different premise: **for a consumer-protection
tool, the errors are not equally costly, so the metric should not treat them as
equal.** A missed unfair clause (false negative) is borne by a consumer who
never reads the ToS and now has false reassurance. A fair clause flagged for
review (false positive) costs a few seconds of a human skim. ToSFlag chooses its
operating point from that asymmetry, and reports honestly what it gives up to
do so.

## Why accuracy is the wrong metric here

The dataset is ~80% fair clauses. A model that predicts "fair" for every clause
scores ~80% accuracy and catches zero unfair clauses - it is useless and looks
respectable. That single fact is why this project does not report accuracy as a
headline number. (Run `scripts/run.py`; the `majority_baseline_acc` column in
the distribution table makes this concrete - it prints ~80 for every split.)

## The thesis, made operational

The decision threshold is chosen to **minimise expected cost**, where a false
negative is weighted `FN_TO_FP_COST_RATIO` times a false positive (default 5:1,
exposed as one constant in `model.py`, not buried). This turns "we care about
recall" - which every project says - into one explicit, inspectable objective.
The central output (`reports/comparison.csv`) puts the recall-first operating
point next to the F1-first point the field defaults to, so the trade is visible:
how much precision was spent, and how many extra clauses go to review, to buy
how much recall.

## What this is and is not

- **Lap 1 (this repo): TF-IDF + Logistic Regression.** Deliberately not a
 transformer. It runs on a CPU in seconds with no model download, so anyone who
 clones the repo reproduces the result. The contribution is the metric decision
 and the task framing, which a heavier encoder would not change. Lap 2 swaps in
 legal-bert behind the same interface *if* the numbers justify the cost.
- It is **not legal advice** and not a substitute for a lawyer. It is a triage
 aid: it surfaces clauses worth a human's attention, ranked by severity.

## Task framing (where this differs from neighbours)

Many "ToS risk" repos actually train on LEDGAR, which is *clause-type*
classification (is this an indemnity clause, a governing-law clause, etc.) and
then attach a ToS label. That is a different task from *unfairness* detection.
ToSFlag runs on `CodeHima/TOS_Dataset` - clauses natively labelled
`clearly_fair` / `potentially_unfair` / `clearly_unfair` - which is the task the
consumer-harm thesis actually describes. The three-class labels are collapsed to
binary (unfair = potentially OR clearly) for the recall head, and kept as an
ordinal severity for ranking flagged clauses.

## What the error analysis found (and what was fixed)

Lap 1 shipped with a deliberate read of its own failures (`reports/error_analysis.md`).
The read drove a concrete fix and then surfaced a deeper finding.

**First read - a featurization gap (fixed).** True misses clustered on ALL-CAPS
warranty and liability disclaimers ("THE SERVICE IS PROVIDED ON AN 'AS IS'
BASIS"). The default TF-IDF lowercases, collapsing "AS IS" into the
near-stopwords "as"/"is" and erasing the signal - even though caps emphasis is a
drafting convention for limitation-of-liability blocks. Adding a caps-preserving
TF-IDF and explicit emphasis features (uppercase ratio, mostly-caps flag) lifted
recall-first test recall from **0.706 to 0.796** (~19 more unfair clauses caught)
and the caps disclaimers dropped out of the miss list entirely.

**Second read - the residual misses are semantic, not lexical.** After the caps
fix, the clauses the model still misses share a different character: waiver-of-
rights and responsibility-shifting clauses written in calm, boilerplate language
("failure to enforce any right shall not constitute a waiver"; "you alone are
responsible for Your Content"). These are unfair because of what the language
*does legally*, not because of any alarming vocabulary - and a bag-of-words model
fundamentally cannot see that. This is the evidenced case for Lap 2: a
legal-domain encoder, not a heavier model in general.

**A note on label noise.** Some "unfair"-labelled items are bare headers
(`NOTICES`, `Violations`, `You will not:`) with no clause content; the pipeline
flags these and reports recall both raw and clause-only. On this dataset the two
numbers are near-identical (0.796 vs 0.797), so header noise is real but is *not*
what caps recall - the model itself is. The dataset also contains subtler,
directional noise: at least one consumer-*protective* carve-out (an arbitration
clause guaranteeing a hearing in the consumer's home county) is labelled unfair.
Both are reasons the Lap 2 cross-check against LexGLUE `unfair_tos` matters.

## The numbers (real, from `scripts/run.py`)

| operating point | recall | precision | F1 | flagged |
|---|---|---|---|---|
| recall-first (5:1 cost) | 0.796 | 0.390 | 0.523 | 0.415 |
| F1-first (the default) | 0.517 | 0.548 | 0.532 | 0.192 |

The recall-first point catches 80% of unfair clauses to the default's 52% - a
28-point recall gain. The cost is explicit and not soft-pedalled: precision falls
to 0.39 and 42% of clauses go to human review. For a consumer-protection triage
aid, that trade is the entire point; for a different use case it would be the
wrong call, which is exactly why the cost ratio is one editable constant.

## Reading the output (no code required)

`scripts/export_report.py` writes a plain per-clause report any reviewer can open
in Excel or Sheets - `reports/clause_report.csv` and a colour-coded
`reports/clause_report.xlsx`. One row per held-out test clause: the clause text,
the dataset's severity label, whether the model flagged it, its risk score, the
outcome (caught / missed / false flag / correct pass), a non-clause marker for
header stubs, and the tokens that drove the flag. It is a triage view, not a new
metric - the model decision shown is the same recall-first operating point above.

```bash
python scripts/export_report.py # writes reports/clause_report.{csv,xlsx}
python scripts/export_report.py --split all # rows for train+val+test, not just test
```

## Honest limitations

- **Single-annotator community dataset.** `CodeHima/TOS_Dataset` (MIT licensed)
 is convenient and correctly shaped, but it is not the peer-reviewed CLAUDETTE
 release and carries one annotator's judgement. Before any production claim,
 cross-check against LexGLUE `unfair_tos` (the standard benchmark) collapsed to
 binary. This cross-check is the planned Lap 2 validation.
- **English only.** Real consumer-protection deployment is multilingual.
- **Sentence/clause level.** No document-level reasoning or clause-to-clause
 interaction is modelled.
- **TF-IDF is shallow.** It keys on surface phrasing ("sole discretion", "we
 reserve the right"). Adversarial rephrasing would evade it - a known weakness
 of this dataset's whole literature, not unique to this model.

## Run it

```bash
pip install -r requirements.txt
python scripts/run.py # uses Hugging Face (needs network)
python scripts/run.py --local data/train.parquet data/validation.parquet data/test.parquet
python scripts/smoke_synthetic.py # verifies mechanics offline (synthetic data)
```

Outputs land in `reports/`: the distribution table, the chosen operating point,
the recall-first-vs-F1-first comparison, and an error analysis listing the
unfair clauses the model still misses (the failures that matter).

## Roadmap

- **Lap 2 (shipped): legal-bert behind the same scorer interface.** `scripts/run_lap2.py`
 fine-tunes (or linear-probes) `nlpaueb/legal-bert-base-uncased` and reuses Lap 1's
 exact operating-point and metric code, so the comparison is apples-to-apples. The
 decisive output is `reports/lap2_recovery.md`: of the unfair clauses TF-IDF missed,
 how many the legal encoder recovers - the direct test of the Lap 1 hypothesis that
 those misses were semantic. The recovery count is reported over real clauses only
 (header/stub label noise excluded, matching `run.py`'s recall discipline). Probe
 mode runs on CPU; `--finetune` is the GPU ceiling.
- **Bring-your-own-ToS inference (the gap between artifact and tool).** Today the
 repo trains on a labelled dataset and reports on that dataset's own held-out test
 split. The model can score arbitrary text, but nothing wired up lets a user feed in
 their own Terms of Service and get a report back. Closing this is two pieces of
 work, only one of them small. The small piece is the inference path: load a cached
 trained model, run the existing pipeline on new text, write the same clause report
 minus the ground-truth columns. The hard piece is clause segmentation: the dataset
 hands clauses pre-split, one per row, but a real ToS is a wall of text, and naive
 splitting breaks on "Inc.", "e.g.", section numbers, and nested sub-clauses. The
 segmentation quality bounds the whole tool's usefulness more than the model does -
 which is exactly why it is named here rather than quietly shipped. A first version
 can sidestep segmentation by accepting an already-split clause list (one per line);
 raw-document segmentation is the harder follow-on.
- Lap 2 (next strands, not yet built): LexGLUE `unfair_tos` cross-check to quantify
 this dataset's label noise against the peer-reviewed benchmark; calibrated
 probabilities (temperature scaling) for the severity ranking.
- Lap 3: FastAPI service + Docker + CI, clause-level review queue output ranked by
 calibrated severity.
