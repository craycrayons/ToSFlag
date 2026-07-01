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

## Built for triage, not verdicts

ToSFlag does a wide, cheap, reproducible first pass: it scores thousands of
clauses in milliseconds on a CPU, for free, deterministically, at a committed
operating point you can inspect and argue with. That is a different job from
rendering a verdict on a single clause. For one clause, a frontier LLM will
often judge it better than this model does - and that is fine, because judging
one clause well is not the job. The job is surfacing the clauses worth a human's
attention across a whole document, or across many documents, at a fixed and
auditable false-negative rate - which is precisely what a non-deterministic,
per-call-priced, un-tunable LLM is the wrong instrument for. The two are not
rivals but stages: a cheap classifier for breadth, an LLM for depth on the small
subset the classifier surfaces. That composition is developed in full below
("The classifier and the LLM are stages, not rivals").

The contribution, then, is not accuracy. It is the judgment about *what to
optimise* in a domain where "correct" is contested - which the rest of this
README makes concrete.

## Why accuracy is the wrong metric here

Start with the shape of the data. In this dataset only about one clause in five
is unfair; the other ~80% are fair. That imbalance is what makes accuracy
misleading, because accuracy just counts how many clauses the model labels
correctly, and on lopsided data that count is easy to inflate without doing
anything useful.

Concretely: a model that ignores the text and labels *every* clause "fair" is
correct on all the fair clauses and wrong only on the unfair ones - so it scores
~80% accuracy while catching zero unfair clauses. It has done no real work, yet
the number looks respectable. Accuracy rewards it for agreeing with the majority
class and hides the fact that it fails completely at the only task that matters:
finding the unfair clauses. (This isn't hypothetical - run `scripts/run.py` and
the `majority_baseline_acc` column prints ~80 for every split, the do-nothing
model's score sitting in your own output.)

So the headline metric cannot be accuracy. And it cannot simply be "recall"
either, because you can trivially get perfect recall by flagging *everything* -
useless in the other direction. What is actually needed is a way to weigh the
two kinds of error against each other, since they are not equally costly: missing
an unfair clause harms a consumer, while over-flagging a fair one costs a few
seconds of review. That asymmetry - not accuracy, not raw recall - is what the
operating point is chosen from, and the next section makes it concrete.

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

## Score your own clauses

The repo ships a frozen model so you can score clauses it has never seen, with
no training and no network. `scripts/train_and_save.py` fits the Lap-1 pipeline
on the cleaned data, derives the committed recall-first threshold on validation
(the same derivation `run.py` uses, so its provenance is real, not a hardcoded
constant), and writes both to `models/` as one unit - the model and the exact
operating point it was chosen at travel together and cannot drift apart. That
frozen model is committed to the repo (~1 MB), so inference needs neither
Hugging Face nor a GPU.

```bash
# one time (or after any retrain): freeze model + operating point
python scripts/train_and_save.py

# score your own clauses - one clause per line
python scripts/infer.py my_clauses.txt
cat my_clauses.txt | python scripts/infer.py          # or piped
```

Input is **one clause per line** (already segmented - see the limitation on
segmentation below). Output is `reports/byo_report.csv` (complete, opens
anywhere) and a colour-coded `reports/byo_report.xlsx` (rows shaded by risk
band, worst first). The report drops the ground-truth columns - `true_label`,
`outcome`, `severity` - because unseen text has no answer key; what remains is
what the model actually produces: `clause`, `flagged`, `risk_score`,
`why_flagged`. This is the *breadth* stage of the two-stage design below.

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

## The classifier and the LLM are stages, not rivals

The obvious question about any classifier in 2026 is "why not just ask an LLM?"
The honest answer is that for judging *one clause*, you often should - and the
project is stronger for saying so. The classifier's value is not depth on a
single clause; it is breadth, cost, and reproducibility across many. The two
belong in the same system, at different stages:

- **Stage 1 - the classifier (this repo).** Recall-first, deterministic, free,
  runs anywhere with no network. It takes *all* clauses and does the wide first
  pass, catching most unfair clauses at a committed, auditable operating point.
- **Stage 2 - an LLM judge (the follow-on).** Depth and explanation, applied
  *only* to the small subset Stage 1 surfaces - including the cases a bag-of-
  words model structurally cannot see (unfairness by accumulated scope, e.g. a
  "perpetual, irrevocable, worldwide, royalty-free" licence whose individual
  words are all neutral). Because it runs on a shortlist, not every clause, its
  per-call cost stays bounded.

The clauses worth an LLM look are the ones where the classifier is *weak*: (a)
the **uncertain band** near the
threshold (in testing, an arbitration clause scored 0.404 against a 0.456
cut-off - it nearly flipped), and (b) **confident blind spots** - low score but
genuinely dangerous (the scope-pileup licence above scored 0.042). No single
rule catches both, so the principled move is to route the *union* of two cheap
signals (score-near-threshold, plus a length/complexity flag or a Lap-1-vs-Lap-2
disagreement) to Stage 2, and keep an agreement column so a human can sort by
the cases where the two stages disagreed - the highest-value review queue.

The sharpest version does not *guess* that band: bin a labelled holdout by score,
measure where the classifier is actually worst, and route by the measurement.

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
- **Bring-your-own-ToS inference (small piece shipped; segmentation still open).**
 The inference path is now built: `scripts/train_and_save.py` freezes the model +
 operating point and `scripts/infer.py` scores an already-split clause list (one
 per line), writing the same report minus the ground-truth columns. See "Score
 your own clauses" above. What remains is the *hard* piece - clause segmentation:
 the dataset hands clauses pre-split, one per row, but a real ToS is a wall of
 text, and naive splitting breaks on "Inc.", "e.g.", section numbers, and nested
 sub-clauses. The segmentation quality bounds the whole tool's usefulness more
 than the model does - which is exactly why it is named here rather than quietly
 shipped. Accepting a pre-split list buys ~80% of the utility; raw-document
 segmentation is the follow-on.
- **Two-stage classifier + LLM (next architecture lap).** The shipped inference
 path is the *breadth* stage. The follow-on adds an LLM *depth* stage on the
 subset the classifier surfaces - routed by where the cheap model is weakest
 (uncertain band + confident blind spots), not by the extremes, with an
 agreement column for human review. Design developed in "The classifier and the
 LLM are stages, not rivals" above. This is the recall-first follow-on that also
 catches the cumulative-scope cases a bag-of-words model structurally misses.
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
