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
 legal-bert behind the same interface and reports what it does and does not buy
 (it trades recall for precision and recovers ~a third of the semantic misses;
 see below), rather than assuming a heavier model is an upgrade.
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
TF-IDF and explicit emphasis features (uppercase ratio, mostly-caps flag) caught
the missed caps disclaimers, which dropped out of the miss list entirely.

**Second read - the residual misses are mostly semantic, not lexical.** After
the caps fix, the clauses the model still misses share a different character:
waiver-of-rights and responsibility-shifting clauses written in calm, boilerplate
language ("failure to enforce any right shall not constitute a waiver"; "you
alone are responsible for Your Content"). These are unfair because of what the
language *does legally*, not because of any alarming vocabulary. A harder tail is
subtler still: clauses that *do* contain trigger vocabulary (arbitration, waiver,
liability) but bury it in syntax a bag-of-words model cannot parse - a
class-action waiver phrased as mutual, an arbitration carve-out worded to sound
even-handed. Both kinds were the predicted case for Lap 2: a legal-domain
encoder that reads structure, not a heavier model in general. (Lap 2 ran, and
partly bore this out - see the legal-BERT section below.)

**A note on label noise (and the cleaning step it forced).** Many
"unfair"-labelled items are bare headers (`NOTICES`, `Terms of Service`, `You
will not:`), section numbers, dates, and nav stubs with no clause content. These
are not fair-or-unfair at all; they are not clauses. An early version flagged
only the most obvious ones; reading the per-clause report directly showed the
filter was leaking, and a strengthened pass removes **~18% of every split** as
structural non-clauses (`data.py`, `drop_nonclauses`, audited row-by-row with
zero real-clause false positives). This matters to the headline number: training
and scoring on the cleaned set drops recall-first recall from the noisy 0.796 to
an honest **0.685**, because the removed stubs were cheap "catches" the model got
credit for. The lower number is the cleanup working, not a regression - and the
held-out LexGLUE cross-check below, run on a dataset with no such noise, confirms
the model itself generalises. The dataset also contains subtler, directional
noise: at least one consumer-*protective* carve-out (an arbitration clause
guaranteeing a hearing in the consumer's home county) is labelled unfair - which
is exactly the kind of thing the LexGLUE cross-check surfaces.

## The numbers (real, from `scripts/run.py`)

| operating point | recall | precision | F1 | flagged |
|---|---|---|---|---|
| recall-first (5:1 cost) | 0.685 | 0.477 | 0.562 | 0.283 |
| F1-first (the default) | 0.554 | 0.574 | 0.564 | 0.190 |

These are the numbers on the cleaned dataset (structural non-clauses removed; see
the label-noise note above). The recall-first point catches 69% of unfair clauses
to the default's 55% - a 13-point recall gain on genuinely hard clauses, with the
free wins from mislabelled stubs no longer inflating either figure. The cost is
explicit and not soft-pedalled: precision is 0.48 and 28% of clauses go to human
review. For a consumer-protection triage aid, that trade is the entire point; for
a different use case it would be the wrong call, which is exactly why the cost
ratio is one editable constant.

## Cross-check: does it generalise? (LexGLUE `unfair_tos`)

The community dataset is one annotator's judgement, so the model is validated
against the peer-reviewed LexGLUE `unfair_tos` benchmark as fully held-out data -
trained on the community set, never shown LexGLUE during training or threshold
selection (`scripts/run_crosscheck.py`, `reports/crosscheck.md`). The same
recall-first threshold transfers almost unchanged (community 0.518 vs a
LexGLUE-tuned 0.537), and recall on the expert benchmark is **0.895** - higher
than on the cleaned community set, because LexGLUE's expert-selected unfair
clauses cluster on lexically-marked legal patterns the model reads well, while
the community set's residual hard cases are the plain-language and
syntactically-buried ones above. Recall by expert category is even (no structural
blind spot; Jurisdiction, Content removal, and Choice of law all at 1.0). Where
the model and the experts disagree, it errs toward flagging - calling unfair some
liability and arbitration clauses the LexGLUE taxonomy leaves unlabelled, which
on a recall-first consumer-protection framing is the correct direction to err.

## Lap 2: does a legal encoder close the semantic gap? (partly)

Lap 1's error analysis predicted the residual misses were semantic, so a
legal-domain encoder should recover them. Lap 2 tests that directly: legal-BERT
(`nlpaueb/legal-bert-base-uncased`) fine-tuned behind the *same* operating-point
and metric code, so the comparison is apples-to-apples (`scripts/run_lap2.py`,
`reports/lap2_recovery.md`). The result is more interesting than a clean win:

| model | recall | precision | F1 | flagged |
|---|---|---|---|---|
| TF-IDF (Lap 1) | 0.685 | 0.477 | 0.562 | 0.283 |
| legal-BERT (Lap 2) | 0.667 | 0.599 | 0.631 | 0.219 |

Legal-BERT recovers **17 of the 53 clauses TF-IDF missed (32%)** - so the
semantic diagnosis was partly right; those clauses really were readable by a
model that encodes legal language. But its overall recall is slightly *lower*,
which means it did not add semantic coverage on top of lexical coverage - it
**traded** one for the other, catching a different slice while dropping some
clauses TF-IDF caught. It buys real precision (0.48 -> 0.60) for that trade. And
~36 of the 53 misses are missed by *both* models - including blatant ones like an
ALL-CAPS "ALL SALES ARE FINAL... no refund" clause - so the residual ceiling is
not "semantic vs surface" at all. Those resist both because they need
clause-level *context* (a venue clause is only unfair relative to where the
consumer lives) or because the label itself is contestable. The honest takeaway:
a heavier model relocates the ceiling rather than breaking it, and the two models
are **complementary** - an ensemble that flags if either fires is the
recall-first follow-on the data points to, not a third model.

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
 release and carries one annotator's judgement. It is cross-checked against
 LexGLUE `unfair_tos` (the standard benchmark) collapsed to binary; the model
 holds 0.895 recall there as held-out data, which is the evidence it generalises
 beyond this one dataset. See the cross-check section above.
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
 how many the legal encoder recovers. Result (see the Lap 2 section above): it
 recovers 32% of the real misses but trades recall for precision rather than
 strictly improving, and a hard residual resists both models - so the two are
 complementary, not ranked. Probe mode runs on CPU; `--finetune` is the GPU run.
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
- **Lap 2 (shipped): LexGLUE `unfair_tos` cross-check.** `scripts/run_crosscheck.py`
 trains on the community set and evaluates on the peer-reviewed benchmark as
 held-out data, reporting generalisation, recall by expert category, and a
 two-way disagreement analysis (`reports/crosscheck.md`). Result summarised in the
 cross-check section above: 0.895 recall on the expert set, threshold transfers
 cleanly, disagreements lean consumer-protective.
- Lap 2 (next strand, not yet built): calibrated probabilities (temperature
 scaling) for the severity ranking, so flagged clauses can be ordered by a
 trustworthy confidence rather than a raw score.
- Lap 3: FastAPI service + Docker + CI, clause-level review queue output ranked by
 calibrated severity.
