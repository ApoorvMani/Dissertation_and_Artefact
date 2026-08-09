# EXPERIMENT_REGISTRY.md

Human-readable companion to `EXPERIMENT_REGISTRY.csv` (37 rows, machine-readable, same evidence). CURRENT (version2) experiments are presented first and are visually dominant, per the audit's presentation rule. All numbers below were independently verified against result artefacts unless marked otherwise; see `evidence_confidence` per row and full paths in `EXPERIMENT_REGISTRY.csv` / `FILE_MAP.md`.

---

# A. Primary dissertation experiments

These are the empirical core of the current dissertation: attack generation → defence matrix → statistical analysis → utility controls → final synthesis, all on the WordPiece victim (`martin-ha/toxic-comment-model`), seed 42 throughout, all `evidence_confidence = HIGH`.

### V2-01 — Clean baseline & source datasets
**Question:** Does the victim correctly classify Jigsaw toxic/benign rows before any attack, and what subset is eligible for attack?
**Result:** Recall 0.624 (156/250 toxic detected); benign FPR 0.032 (8/250). 156 eligible toxic rows carried through every downstream attack.
**Scripts → outputs:** `02_build_datasets.py`, `02_clean_baseline.py` → `data/clean_baseline_{toxic,benign}.csv`.

### V2-02 — TokenBreak attack
**Question:** Does TokenBreak (BreakPrompt Algorithm 1) fool the victim on the 156 eligible rows?
**Result:** 150/156 successful, **ASR 96.2%**, mean 9.2 words modified, mean confidence drop 0.8081.
**Scripts → outputs:** `04_tokenbreak_attack.py` → `data/tokenbreak_results.csv` (156×26).
**Supersedes:** V2-11 (old ad hoc heuristic implementation).

### V2-03 — AdvTok attack
**Question:** Does AdvTok (noncanonical WordPiece search) fool the victim while preserving visible text exactly?
**Result:** 150/156 successful, **ASR 96.2%**; seed-alone accounted for 82/156 (52.6%), local search added the remaining 68.
**Scripts → outputs:** `05_advtok_attack.py` → `data/advtok_results.csv` (156×31), `data/advtok_successes.csv` (150×31).
**Supersedes:** V2-12 (pre-bugfix run — 3 documented correctness bugs).

### V2-04 — Unicode attack families
**Question:** Do 4 Unicode sub-attacks (invisible/homoglyph/compat/reorder) fool the victim?
**Result:** compatibility 91.0% (142/156), reorder 88.5% (138/156), homoglyph 85.9% (134/156), **invisible 0.0% (0/156)**.
**Scripts → outputs:** `07_unicode_attacks.py` → `data/unicode_{invisible,homoglyph,compat,reorder}.csv` (156 each), `unicode_results.csv` (624, verified concatenation).

### V2-05 — Seven-defence matrix
**Question:** How does each of 7 defences perform against each of 6 attacks, paired clean-vs-attacked?
**Result:** 7 distinct defences confirmed (tokenizer translation, canonical reject, canonical replace, unicode sanitiser, NFKC+confusables, global CPT, windowed CPT) × 6 distinct attacks = 42 cells, 6,552 paired rows.
**Scripts → outputs:** `08_defences.py`, `09_defence_matrix.py` → `data/defence_matrix.csv` (6,552×23), `data/defence_matrix_summary.csv`.
**Supersedes:** V2-13 (old unpaired evaluation methodology).

> **Post-hoc design rationale** (added 2026-08-08 by this audit at the user's request — reconstructed from the current script's logic, not found in any dated project note file):
> The OLD (superseded, V2-13) unpaired matrix risked attributing a defence's own side-effect to the attacker — if a defence changed a clean toxic sample to non-toxic, the attacked version would also read non-toxic and could be misread as attack success, when the defence, not the attacker, caused the flip.
> The FINAL (paired, this experiment) design fixes this: it first requires the clean sample remain toxic under the *same* defence, and only then asks whether the attacked version becomes non-toxic under that defence — so attack success is never confounded with defence-induced classifier degradation.

### V2-06 — Paired statistical analysis (step10)
**Question:** What are the bootstrap-CI'd effectiveness/utility/latency rates per attack×defence cell?
**Result:** TokenBreak best defended by tokenizer translation (95.3%→46.5% paired ASR, 95% CI 40.2–57.5pp reduction); AdvTok fully neutralised (→0%) by five of seven defences; **one HARMFUL defence-induced regression**: tokenizer translation turns Unicode-invisible from 0% ASR into 13.4% ASR.
**Scripts → outputs:** `10_analysis.py` → `results/step10/*` (attack_metrics, clean_controls 36,750 rows, detection_metrics, 10 figure pairs, `report.html`).

### V2-07 — Complex legitimate-input utility
**Question:** What is the collateral-damage/false-trigger cost of each defence on complex-but-legitimate text?
**Result:** Global CPT blocks/triggers on 55.2% of complex legitimate inputs (highest cost of any defence); tokenizer translation preserves 98.7% of decisions on the same set.
**Scripts → outputs:** `11_complex_utility.py` → `data/complex_legitimate_600.csv` (600, one row per source example), `results/step11/*`.

### V2-08 — Final coverage matrix & margin-stratified ASR
**Question:** Which attack×defence cells achieve complete/partial/no-effect coverage, and does attack success track the victim's confidence margin?
**Result:** Coverage status: 19 NO CLEAR EFFECT / 9 COMPLETE / 7 PARTIAL / 6 NO BASE ATTACK / 1 HARMFUL. On high-confidence rows (P(toxic)>0.80): TokenBreak/AdvTok 96.1% ASR, Unicode compat 89.8%, reorder 87.5%, homoglyph 84.4%, invisible 0.0%.
**Scripts → outputs:** `13_freeze_current_results.py` → `results/final_current/*` (this is the dissertation's headline summary artefact).

---

# B. Supporting current experiments

### V2-09 — Cross-tokenizer BPE/Unigram clean-baseline (generalisation groundwork)
**Status: SUPPORTING, not proof of attack generalisation.**
**Question:** Do BPE (RoBERTa) and Unigram (DeBERTa-v3) classifiers agree with the WordPiece victim on *clean, unattacked* Jigsaw rows?
**Result:** WordPiece eligible 156/250 (recall 0.624); BPE eligible 214/250 (recall 0.856); Unigram eligible 242/250 (recall 0.968); shared-eligible-across-all-three 155/250.
**Important:** No attack was run against the BPE or Unigram models here. This establishes eligible sets only. Do not cite as evidence attacks transfer across tokenizer families — see Section E, V2-14.
**Scripts → outputs:** `12_cross_tokenizer_baseline.py` → `data/cross_tokenizer_metadata.json` and related CSVs.

---

# C. Preliminary/historical research programme (deftok5)

**This entire section demonstrates research development, statistical/reproducibility rigor, and substantial prior work. It is presented for methodological completeness and transparency about the dissertation's evolution. None of the numbers in this section are pooled with, added to, or compared directly against Section A/B's attack×defence results — different victim model, different attack implementations, different corpora, and a different research question (cross-tokenizer disagreement as a *detector*, not attack success against a fixed defence set).**

The programme ran entirely 2026-07-15 through 2026-07-31 (git-tracked, 46 commits) and pursued one central question: can cross-tokenizer boundary disagreement (Jaccard divergence between GPT-2/BERT/XLNet) detect adversarial or non-canonical text, benchmarked against a CPT (characters-per-token) baseline?

**Mechanics & metric definition (PRELIMINARY, pedagogical):** D5-01 (tokenization-mechanics demo), D5-02 (metric definition, hand-worked), D5-03 (whitespace-projection demo). Single hand-picked examples, not corpus statistics.

**Corpus-scale reconciliation & first correction (PRELIMINARY / SUPERSEDED):** D5-04 established the corpus-scale same-family BERT-XLNet Jaccard mean at ~0.07 (100- and 10k-sentence WikiText-2), correcting a prior "0.763" value (D5-05, SUPERSEDED) that was traced to a different, mislabelled configuration.

**Attack generator demos (PRELIMINARY, toy implementations):** D5-06 — clean-room TokenBreak/AdvTok mechanism demos, explicitly logged as fidelity deviations from the source papers (word-length-proxied saliency; single distance-1 split). Not the same code as version2's V2-02/V2-03.

**Real-traffic corpus acquisition and freezing (SUPPORTING — reusable artefacts):** D5-07 (one shard each of LMSYS-Chat-1M, WildChat-1M, ToxicChat, sha256-pinned); D5-08/D5-09 (pre-registered redaction-artefact tests on LMSYS and WildChat — both **partially refuted**, with a direction mismatch between the two corpora that was left explicitly unresolved); D5-10 (**SUPERSEDED** — an initial "0.1% FPR is estimable" scale claim was retracted once conversations, not rows, were recognised as the correct resampling unit); D5-11 (placeholder-mechanism constructed test, PRELIMINARY, inconclusive/mixed); D5-12 (ToxicChat harmful-but-unobfuscated control, SUPPORTING — a genuine negative/bounded result: harmful text was *not* more divergent than benign text on the BERT-XLNet pair).

**Detector/analysis tooling (SUPPORTING — reusable methodology, not empirical claims):** D5-13 (ROC/AUC, operating-points library), D5-14 (CPT baseline characterisation — **not** the source of version2's independently-calibrated CPT thresholds), D5-15 (GPT-2 perplexity baseline characterisation; also a reproducibility-incident case study — a missing `.to(DEVICE)` call silently ran early scoring on CPU), D5-16 (bootstrap/clustering/degeneracy/residualisation tooling; also documents a 25.69% duplicate-template contamination finding in LMSYS).

**Attack corpus at scale (SUPPORTING):** D5-18 — a 293,034-row paired, multi-seed TokenBreak attack corpus (confirms the "293k" figure referenced in the audit brief), with an accompanying disclosure (D5-19, **SUPERSEDED**) that a previously logged "~14.5% cross-seed mismatch" figure had never actually been measured against any running code.

**Reproducibility incident (SUPPORTING — case study):** D5-20 — an overnight batch run (2026-07-31) discovered that prior results had been contaminated by a global WinPython interpreter instead of the project's pinned `.venv`; 11+ result files were quarantined, `cpt_xlnet` was voided, and an interpreter-identity guard (`envguard.py`) was built and retrofitted repo-wide in direct response.

**Terminal confirmatory result (PRELIMINARY — the programme's core, negative finding):** D5-21 — the pre-registered E1 test of cross-tokenizer divergence (BERT–XLNet Jaccard) vs. a single-tokenizer CPT baseline (GPT-2), decided by a paired bootstrap 95% CI on ΔTPR@1%FPR on LMSYS:

| | jac_bert_xlnet | cpt_gpt2 |
|---|---|---|
| TPR@1%FPR | 1.0950% | 1.0701% |
| AUC (secondary) | 0.577678 | 0.521741 |

ΔTPR = 0.000250, 95% CI = **[-0.001594, 0.002125]** (includes 0) → **decision: NOT SUPERIOR**. This is the deftok5 detector idea's documented terminal result. version2's own abandonment of `06_detector_scores.py` ("cross-tokenizer Jaccard divergence... my own idea and not paper-backed") is consistent with this outcome.

**Unresolved follow-on (FUTURE — see Section E):** D5-22 — E2 (position sensitivity) is pre-registered only; no result exists in the repository.

---

# D. Superseded/corrected experiments

| id | repo | what was superseded | superseded by |
|---|---|---|---|
| V2-11 | version2 | Ad hoc heuristic TokenBreak implementation (`backup/04_tokenbreak_attack_old.py`) | V2-02 (paper-faithful BreakPrompt Algorithm 1 rewrite) |
| V2-12 | version2 | Pre-bugfix AdvTok run (`advtok_results_old.csv`) — 3 documented bugs (parity check, validity check, segmentation logic) | V2-03 |
| V2-13 | version2 | Unpaired defence-matrix methodology (`backup/09_defence_matrix_old.py`) | V2-05 (paired clean-vs-attacked methodology). **No contemporaneous note explains this change; a post-hoc design rationale was added 2026-08-08 by this audit — see V2-05 above. MEDIUM confidence, since the rationale is reconstructed, not sourced from a dated project note.** |
| D5-05 | deftok5 | "0.763" same-family BERT-XLNet Jaccard answer-key value (mislabelled, wrong configuration) | D5-04 (~0.07, corpus-scale) |
| D5-10 | deftok5 | "134,217 rows clears 10^5" / "0.1% FPR estimable" scale claim (commit `bd91a9b`) | commit `fa00ea3` — conversations (85,246), not rows, are the correct unit; 0.1% FPR unachievable |
| D5-19 | deftok5 | "~14.5% cross-seed mismatch" limitation — never actually measured against running code | D5-18's measured 0.0000 cross-seed agreement (a different quantity, not a direct replacement) |

---

# E. Planned but not executed work

| id | repo | what was planned | evidence it did not run |
|---|---|---|---|
| **V2-14** | version2 | **Attack generalisation across BPE/Unigram tokenizer families** (TokenBreak/Unicode/AdvTok vs. RoBERTa/DeBERTa) | `backup/13_cross_tokenizer_string_attacks.py` and `backup/13b_cross_tokenizer_advtok.py` are drafted, argparse-ready, declare specific output filenames — **none of those output files exist anywhere in the repository**. `notes/pipeline.md`'s own results table is filled entirely with `?` placeholders. **Do not cite "attacks generalise across tokenizer families" as an established finding.** |
| V2-15 | version2 | Guard-model baseline defence (an 8th defence) | Planned in `notes/to-do` and `notes/handoffs.md`, absent from the executed 7-defence set with no documented reason for dropping it. |
| V2-16 | version2 | Applying current defences to large-scale LMSYS/WildChat traffic | Listed under a "NEXT" (not "NOW"/"DONE") heading in `notes/pipeline.md`; no work started. |
| D5-22 | deftok5 | E2 — position sensitivity of cross-tokenizer divergence (suffix/mid/double-first vs. front) | Pre-registration (`prereg/E2.md`) and implementation code exist (commit `509db61`); repo HEAD (`95420f0`) is the pre-reg commit itself; **no `results/E2_results.md` or equivalent exists**. |

---

*Full per-row evidence, exact file paths, and all numeric fields (n_source, n_evaluated, n_generated_rows, seeds, statistics) are in `EXPERIMENT_REGISTRY.csv`. Cross-reference `FILE_MAP.md` for script/artefact inventories and `AUDIT_WARNINGS.md` for unresolved discrepancies.*
