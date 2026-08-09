# Final publication figure notes

These figures are generated from frozen result artefacts only.

No attack, defence, model inference, calibration or statistical analysis
is rerun by this script.

## Main-body recommendation

1. fig01_experimental_design
2. fig02_undefended_attack_asr
3. fig03_attack_defence_coverage
4. fig04_tokenbreak_paired_reduction
5. fig05_complex_utility
6. fig06_margin_stratified_asr
7. fig07_defence_induced_regression

## Appendix recommendation

- figA1_advtok_action_audit
- figA2_model_tokenizer_baseline
- figA3_research_evolution

## Important interpretation rules

Figure 3:
"COMPLETE" means defended ASR reached zero in the evaluated setting.
It must not be written as "the attack is solved".

Figure A1:
This is an implementation/audit figure only. The recorded actions are
not presented as a novel security mechanism or contribution.

Figure A2:
Model and tokenizer both differ. The eligibility differences therefore
cannot be attributed to tokenizer family alone.

Figure A3:
Historical results are evidence of research development and are not
pooled with the primary attack-defence experiment.

## File format

Use PDF or SVG in the final dissertation whenever possible.

PNG is included only as a preview or fallback.

## Print check

Before submission:

- insert figures at their real dissertation width;
- export the dissertation to PDF;
- inspect at 100% zoom;
- print/export one page in grayscale;
- verify all labels and numbers remain readable;
- confirm that no conclusion depends on colour alone.
