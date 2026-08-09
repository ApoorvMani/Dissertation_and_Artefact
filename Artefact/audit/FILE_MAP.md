# FILE_MAP.md

Evidence-backed map of every important script/result. CURRENT (version2) files first, then HISTORICAL (deftok5) files. All paths relative to each repository's root unless given in full. `evidence_confidence` reflects how directly this audit verified the row (HIGH = artefact read/inspected directly; MEDIUM = documented in notes but artefact not independently opened; LOW = inferred from filename/comment only).

---

## CURRENT — `D:\Study\NCL\Dissertation\artefact\version2`

### Pipeline scripts

| file | what it does | inputs | outputs | experiment | status | confidence |
|---|---|---|---|---|---|---|
| `01_download_assets.py` | Downloads/caches victim (`martin-ha/toxic-comment-model`, WordPiece) + 3 reference tokenizers | HF Hub | none persisted (cache only) | V2-00 setup | PRIMARY | HIGH |
| `02_build_datasets.py` | Freezes Jigsaw (250 toxic + 250 benign) and WikiText (5000 calib + 5000 heldout), seed 42 | HF `jigsaw-toxic-comment-classification-challenge`, `Salesforce/wikitext` | `data/jigsaw_toxic_250.csv`, `data/jigsaw_benign_250.csv`, `data/wikitext_calibration_5000.csv`, `data/wikitext_heldout_5000.csv` | V2-01 | PRIMARY | HIGH |
| `02_clean_baseline.py` (notes call it "Step 3" — filename/label mismatch, unexplained) | Scores victim on all 500 Jigsaw rows; defines eligible attack set | `jigsaw_toxic_250.csv`, `jigsaw_benign_250.csv` | `data/clean_baseline_toxic.csv`, `data/clean_baseline_benign.csv` | V2-01 | PRIMARY | HIGH |
| `06_detector_scores.py` | Cross-tokenizer Jaccard divergence detector — **explicitly abandoned** ("my own idea and not paper-backed") | n/a | none | V2-10 (abandoned) | SUPERSEDED | HIGH |
| `04_tokenbreak_attack.py` | TokenBreak / BreakPrompt Algorithm 1 reimplementation | `clean_baseline_toxic.csv` (156 eligible) | `data/tokenbreak_results.csv` (156×26) | V2-02 | PRIMARY | HIGH |
| `05_advtok_attack.py` | AdvTok (Geh, Shao & Van den Broeck, ACL 2025) adapted to WordPiece | `clean_baseline_toxic.csv` (156 eligible) | `data/advtok_results.csv` (156×31), `data/advtok_successes.csv` (150×31) | V2-03 | PRIMARY | HIGH |
| `07_unicode_attacks.py` | 4 Unicode sub-attacks (invisible/homoglyph/compat/reorder), Boucher et al. 2022-style DE search; `compat` is an in-code-flagged own extension | `clean_baseline_toxic.csv`, `data/intentional.txt` | `data/unicode_{invisible,homoglyph,compat,reorder}.csv` (156×27 each), `data/unicode_results.csv` (624×27) | V2-04 | PRIMARY | HIGH |
| `08_defences.py` | 7-defence library (D1–D7) + CPT threshold calibration on clean WikiText only | `wikitext_calibration_5000.csv`, `intentional.txt` | `data/cpt_thresholds.json` (`{"global":2.94093137254902,"window":1.1}`) | V2-05 | PRIMARY | HIGH |
| `09_defence_matrix.py` | Paired clean-vs-attacked evaluation, 6 attacks × 7 defences | `tokenbreak_results.csv`, `advtok_results.csv`, `unicode_results.csv` | `data/defence_matrix.csv` (6,552×23), `data/defence_matrix_summary.csv` (42×16) | V2-05 | PRIMARY | HIGH |
| `10_analysis.py` | Bootstrap CIs (N=5000), clean controls, figures, HTML report; does not rerun attacks | `defence_matrix.csv`, `wikitext_heldout_5000.csv`, `jigsaw_benign_250.csv`, `cpt_thresholds.json` | `results/step10/*.csv`, `results/step10/figures/*`, `results/step10/report.html` | V2-06 | PRIMARY | HIGH |
| `11_complex_utility.py` | 600-row complex-legitimate-input utility test (6×100 categories) across 7 defences | CodeSearchNet, GoEmotions, FLORES, own Jigsaw-benign rows | `data/complex_legitimate_600.csv` (600×9), `results/step11/*.csv`, `results/step11/figures/*` | V2-07 | PRIMARY | HIGH |
| `12_cross_tokenizer_baseline.py` | Clean-baseline-only concordance for BPE (RoBERTa) and Unigram (DeBERTa) models — **no attacks run** | `jigsaw_toxic_250.csv`, `jigsaw_benign_250.csv`, `clean_baseline_toxic.csv` | `data/cross_bpe_clean_{toxic,benign}.csv`, `data/cross_unigram_clean_{toxic,benign}.csv`, `data/cross_tokenizer_eligibility.csv`, `data/cross_tokenizer_shared_eligible.csv`, `data/cross_tokenizer_metadata.json` | V2-09 | SUPPORTING (generalisation groundwork, not attack-generalisation proof) | HIGH |
| `13_freeze_current_results.py` | Final snapshot: coverage matrix, margin-stratified ASR; no model calls, no new attacks | step10/step11 outputs, `defence_matrix.csv` | `results/final_current/coverage_matrix.csv`, `margin_overall.csv`, `margin_stratified_asr.csv`, `headline_summary.txt`, `results/final_current/figures/*` | V2-08 | PRIMARY | HIGH |
| `attack_schema.py` | Shared row schema (`build_record`) used by 04/05/07 | n/a (library) | n/a | supporting all above | PRIMARY (infra) | HIGH |
| `test.py` | 137-byte file, minor/incidental | — | — | n/a | UNKNOWN role | LOW |

### Backup / superseded / unexecuted scripts (`backup/`)

| file | what it does | status | confidence |
|---|---|---|---|
| `04_tokenbreak_attack_old.py` | Ad hoc "10 most influential words" heuristic attack — pre-paper-fidelity-rewrite | SUPERSEDED by `04_tokenbreak_attack.py` | HIGH |
| `08_defences_old.py` | Byte-identical duplicate of `08_defences.py` at an earlier point (md5-identical to another backup copy, differs from live root file only in docstring/comment expansion) | SUPERSEDED (redundant copy) | HIGH |
| `09_defence_matrix_old.py` | Unpaired attack/defence evaluation (no clean-vs-attacked pairing) | SUPERSEDED by `09_defence_matrix.py`; reason for the change not documented in notes | HIGH (change exists) / LOW (stated reason) |
| `13_cross_tokenizer_string_attacks.py` | Drafted: TokenBreak + 4 Unicode attacks vs BPE/Unigram models | **FUTURE — no output evidence found anywhere in repo** | HIGH (non-execution) |
| `13b_cross_tokenizer_advtok.py` | Drafted: AdvTok generalised to BPE-internal / Unigram-internal segmentations | **FUTURE — no output evidence found anywhere in repo** | HIGH (non-execution) |
| `advtok_results_old.csv` | Pre-bugfix AdvTok output (156×15, narrow pre-refactor schema) | SUPERSEDED by `data/advtok_results.csv` (bug fixes: parity check, validity check, segmentation logic) | HIGH |
| `tokenbreak_results_old.csv` (backup) / `old_tokenbreak_results.csv` | Pre-rewrite TokenBreak output (heuristic methodology) | SUPERSEDED by `data/tokenbreak_results.csv` | HIGH |
| `defence_matrix.csv` (backup) | Unpaired-methodology defence matrix, different byte size than current | SUPERSEDED by `data/defence_matrix.csv` | HIGH |

### Data artefacts (`data/`)

| file | rows×cols (verified) | belongs to | confidence |
|---|---|---|---|
| `jigsaw_toxic_250.csv` / `jigsaw_benign_250.csv` | 250×4 each (raw `wc -l` inflated by embedded newlines in text field — not extra rows) | V2-01 | HIGH |
| `wikitext_calibration_5000.csv` / `wikitext_heldout_5000.csv` | 5000×3 each | V2-01/V2-06 | HIGH |
| `clean_baseline_toxic.csv` / `clean_baseline_benign.csv` | 250×6 each (superset of jigsaw files + 2 scored columns) | V2-01 | HIGH |
| `tokenbreak_results.csv` | 156×26 | V2-02 | HIGH |
| `tokenbreak_results_old.csv` | 156×12 (pre-refactor schema) | superseded | HIGH |
| `advtok_results.csv` | 156×31 | V2-03 | HIGH |
| `advtok_results_old.csv` | 156×15 (pre-bugfix schema) | superseded | HIGH |
| `advtok_successes.csv` | 150×31 (subset, `attack_success==1`) | V2-03 | HIGH |
| `unicode_invisible.csv` / `unicode_homoglyph.csv` / `unicode_compat.csv` / `unicode_reorder.csv` | 156×27 each | V2-04 | HIGH |
| `unicode_results.csv` | 624×27 (= 156×4 concatenation, verified) | V2-04 | HIGH |
| `defence_matrix.csv` | 6,552×23 (= 936 attack instances × 7 defences) | V2-05 | HIGH |
| `defence_matrix_summary.csv` | 42×16 (6 attacks × 7 defences) | V2-05 | HIGH |
| `complex_legitimate_600.csv` | 600×9 (raw `wc -l`=2774 is embedded-newline artefact from code snippets, not extra rows) | V2-07 | HIGH |
| `cpt_thresholds.json` | `{"global": 2.94093137254902, "window": 1.1}` | V2-05 | HIGH |
| `cross_bpe_clean_toxic.csv` / `cross_bpe_clean_benign.csv` / `cross_unigram_clean_toxic.csv` / `cross_unigram_clean_benign.csv` | 250×8 each | V2-09 | HIGH |
| `cross_tokenizer_eligibility.csv` | 250×5 | V2-09 | HIGH |
| `cross_tokenizer_shared_eligible.csv` | 155×4 | V2-09 | HIGH |
| `cross_tokenizer_metadata.json` | WordPiece 156/250 (0.624), BPE 214/250 (0.856), Unigram 242/250 (0.968), shared-all-three 155/250 | V2-09 | HIGH |
| `intentional.txt` | UTS #39 confusables table (used by unicode + defence scripts) | V2-04/V2-05 | HIGH |

### Results

| file/dir | belongs to | key content | confidence |
|---|---|---|---|
| `results/step10/` (`attack_metrics.csv` 42×31, `clean_controls.csv` 36,750×15, `clean_summary.csv` 14×18, `detection_metrics.csv` 42×12, 10 figure pairs, `report.html`, `literature_notes.txt`) | V2-06 | Defended ASR, detection/block/re-encode rates, clean-control retention, HARMFUL regression flagged (Unicode invisible + tokenizer translation) | HIGH |
| `results/step11/` (`complex_utility_overall.csv` 7×.., `complex_utility_summary.csv` 42×27, `complex_utility_rows.csv`, 8 figure pairs, `report.html`, `dataset_provenance.txt`) | V2-07 | Per-defence trigger/block/operational-preservation rates on 600 complex legitimate rows | HIGH |
| `results/final_current/` (`coverage_matrix.csv` 42×.., `margin_overall.csv` 6×.., `margin_stratified_asr.csv` 18×.., `headline_summary.txt`, 2 figure pairs) | V2-08 | Coverage-status counts (19 NO CLEAR EFFECT / 9 COMPLETE / 7 PARTIAL / 6 NO BASE ATTACK / 1 HARMFUL); margin-stratified ASR by confidence bin | HIGH |

### Notes / documentation

| file | content | confidence |
|---|---|---|
| `notes/handoffs.md` | Session-to-session handoff log: fixed setup table, per-step FINAL results, "BUGS FOUND AND FIXED" list, open items | HIGH |
| `notes/pipeline.md` | Plain-English pipeline walkthrough incl. the Step-13 "?"-filled future table | HIGH |
| `notes/observations.md` | Findings/observations, cross-consistent with results files | HIGH |
| `notes/to-do` | Early Day-1/Day-2 planning transcript; step numbering does not match final 01–13 structure; mostly superseded except one still-relevant truncation-check item | MEDIUM |
| `notes/claudecode.md` | **Not about version2** — pasted output of an unrelated audit of 8 sibling `deftok*` folders | HIGH (content), flagged as out-of-scope anomaly |
| `requirements.txt` | Python dependency pins | HIGH |

---

## HISTORICAL — `D:\Study\NCL\Coding\deftok5`

### Core library (`deftok5/` package)

| file | implements | confidence |
|---|---|---|
| `deftok5/analysis/bootstrap.py` | BCa bootstrap CIs, unit-resampled + paired-difference | HIGH |
| `deftok5/analysis/clustering.py` | ICC(1), design effect, effective sample size, resampling-unit union-find | HIGH |
| `deftok5/analysis/degeneracy.py` | Tie-profile / structural-saturation guard against meaningless AUC=1.0 comparisons | HIGH |
| `deftok5/analysis/residualise.py` | Fertility computation, confound residualisation, nested OOF AUC | HIGH |
| `deftok5/attacks/tokenbreak_corpus.py` | Paired multi-seed TokenBreak-style attack-corpus builder (4 position variants); fertility-ranked target selection explicitly flagged as a fidelity deviation from the source paper | HIGH |
| `deftok5/detectors/cpt.py` | Characters-per-token baseline detector | HIGH |
| `deftok5/detectors/perplexity.py` | GPT-2 perplexity baseline (Jain et al.) | HIGH |
| `deftok5/metrics/roc.py` | ROC/AUC from scratch (Mann-Whitney U, midrank ties) | HIGH |
| `deftok5/metrics/operating_points.py` | Low-FPR threshold/TPR computation | HIGH |
| `deftok5/envguard.py` | Interpreter-identity guard — built in direct response to the WinPython contamination incident | HIGH |
| `deftok5/prereg.py` | Write-before-run pre-registration enforcement | HIGH |

### Pipeline / analysis scripts (`scripts/` + root)

| file | purpose | outputs | confidence |
|---|---|---|---|
| `fetch_{lmsys,wildchat,toxicchat}.py` | Download one shard each of raw corpora | `data/raw/{lmsys,wildchat,toxicchat}/` | HIGH |
| `extract_{lmsys,wildchat,toxicchat}.py` | Extract English user turns | `data/processed/{lmsys,wildchat}_english_user_turns.parquet`, `toxicchat_user_turns.parquet` | HIGH |
| `freeze_{lmsys,wildchat,toxicchat}_null.py` | Apply pre-registered redaction drop, freeze benign-null corpora | `lmsys_null_frozen.parquet` (134,217 turns/85,246 convs), `wildchat_null_frozen.parquet` (75,839/29,223), `toxicchat_frozen.parquet` (10,165, degenerate clustering) | HIGH |
| `score_{redaction,wildchat,toxicchat}.py` | 3-way Jaccard scoring | `{lmsys,wildchat,toxicchat}_divergence_scores.parquet` | HIGH |
| `filter_*_scores_to_frozen.py` | Restrict scores to frozen-null keys | `{lmsys,wildchat,toxicchat}_null_scores.parquet` / `toxicchat_scores.parquet` | HIGH |
| `scripts/build_attack_corpus.py` | E0: paired multi-seed TokenBreak attack corpus, floor=200 chars, seeds 0–4 | `data/processed/attack_corpus_paired.parquet` (293,034 rows) + `.md` | HIGH |
| `scripts/score_attack_corpus.py` | E0 scoring: Jaccard + CPT×3 + perplexity, GPU batch=1 | `data/processed/attack_scores.parquet` (293,034×12) + `.md` | HIGH |
| `scripts/e1_analysis.py` | E1 confirmatory analysis (gated by `prereg.assert_prereg_ready`) | `results/E1_results.md` + envfingerprint/provenance sidecars, `figures/E1_logroc.png` | HIGH |
| `scripts/e1_unit_sensitivity.py` | Exploratory CI-sensitivity check (not pre-registered) | Part B of `E1_results.md` (provenance of exact write path not fully traced) | MEDIUM |
| `scripts/characterise_cpt_null.py` / `characterise_perplexity_null.py` / `characterise_clustering.py` / `characterise_fertility_confound.py` | Benign-null characterisation | `data/processed/*_characterisation.parquet` + `.md` | HIGH |
| `scripts/preregister.py` | CLI wrapper for `deftok5.prereg.write_prereg` | `prereg/<id>.md` | HIGH |
| `placeholder_mechanism.py` | Pre-registered constructed-input test (5 strings × 4 conditions) | stdout only (result quoted in decisions.md) | MEDIUM |
| `scripts/recheck_attack_scores.py`, `scripts/recheck_jaccard_null.py`, `_secA_text_stability.py`, `_secB_divergence_audit.py`, `scripts/scratch/*`, `part2_*`, `part3_*`, `diag_5rows_both_interp.py` | Ad hoc forensic scripts investigating the WinPython contamination | stdout / scratch JSON only, self-declared diagnostic-only | HIGH (existence) |

### Results / provenance

| file | content | confidence |
|---|---|---|
| `results/E1_results.md` + `.envfingerprint.json` + `.provenance.json` | E1 confirmatory result: TPR@1%FPR jac_bert_xlnet=0.010950 vs cpt_gpt2=0.010701, ΔTPR CI=[-0.001594, 0.002125], **decision NOT SUPERIOR** | HIGH |
| `figures/E1_logroc.png` | Log-ROC figure accompanying E1 | HIGH (exists), image content not inspected |
| `prereg/E1.md` | Pre-registration for E1 (commit `f323c5a`) | HIGH |
| `prereg/E2.md` | Pre-registration for E2 (commit `95420f0`, repo HEAD) — **no corresponding result file exists** | HIGH |
| `data/processed/*.md` (5 characterisation summaries) | Raw percentile/correlation tables; each ends with an unfilled "WHAT WE FOUND" placeholder | HIGH |
| `data/processed/quarantine_wrong_venv/`, `quarantine_no_sidecar/`, `quarantine_scratch_provenance/` | Contaminated/superseded artefacts from the WinPython incident and pre-envguard runs | HIGH |
| `logs/overnight_20260731T014033Z.log`, `logs/overnight_summary.md` | J1–J3 overnight batch run log; WinPython quarantine, orchestrator path bug + repair, final job statuses (all COMPLETE) | HIGH |
| `decisions.md` (107KB), `progress.md` (64KB), `experiments.md` | Full decision/progress log across all phases A0→E2 | HIGH |

### Tests

`tests/test_{bootstrap,clustering,cpt,degeneracy,envguard,operating_points,perplexity,prereg,residualise,roc,tokenbreak_corpus}.py` — unit tests for the library above; `test_envguard.py` and `test_prereg.py` specifically guard against repeats of the WinPython and pre-registration-discipline incidents. HIGH confidence (all files confirmed present and read in part).
