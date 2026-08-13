# Dissertation and Artefact

This repository contains a dissertation and its accompanying research artefact investigating tokenizer-level adversarial attacks against toxic-content classifiers, and the effectiveness of defences against them.

## Contents

- `Dissertation_250593365.pdf` — the full dissertation write-up.
- `Artefact/` — the experimental pipeline and results supporting the dissertation.

## Artefact overview

The artefact implements and evaluates a pipeline of tokenizer-based adversarial attacks (e.g. TokenBreak, AdvTok, Unicode-based attacks) against a WordPiece-tokenized toxic-comment classifier, along with a library of candidate defences and cross-tokenizer generalisation checks.

Key components:

- `01_download_assets.py` – `14_freeze_reproducibility.py` — the numbered pipeline scripts, run roughly in order, covering dataset construction, attack implementation, defence evaluation, and results analysis.
- `attack_schema.py` — shared record schema used by the attack scripts.
- `data/` — frozen datasets and per-attack result CSVs/JSON produced by the pipeline.
- `results/` — analysis outputs, figures, and HTML reports generated from the data.
- `figures/` — final figures used in the dissertation.
- `audit/` — provenance documentation (`EXPERIMENT_REGISTRY.md`, `FILE_MAP.md`) describing what each script/artefact does and its verification status.
- `requirements.txt` — Python dependencies for reproducing the pipeline.

## Reproducing

```
pip install -r Artefact/requirements.txt
```

Then run the numbered scripts in `Artefact/` in order. See `Artefact/audit/FILE_MAP.md` for a detailed breakdown of each script's inputs, outputs, and status.
